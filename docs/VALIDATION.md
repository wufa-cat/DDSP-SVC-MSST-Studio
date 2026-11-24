# 实测报告（本分支交付前跑过的验证）

## 测试机

| 项 | 值 |
|---|---|
| 系统 | Windows Server 2022 Datacenter 10.0.20348 |
| CPU | AMD EPYC 2 核 4 线程 |
| 内存 | 8 GB |
| 显卡 | **无**（`nvidia-smi` 报驱动未安装，`torch.cuda.is_available()=False`） |
| 磁盘 | C: 107 GB |

因此本报告验证的是 **CPU 回落路径**；GPU 路径验证的是「环境在 cu128 下能正确构建/导入」
（`torch.__version__ = 2.7.1+cu128`，DLL 完整，真实显卡机器上即为 GPU 执行）。

## 1. 环境

```
torch 2.7.1+cu128 / torchvision 0.22.1+cu128 / torchaudio 2.7.1+cu128
python 3.10.22
24 个核心模块 import 全通过（librosa/soundfile/pyworld/parselmouth/torchcrepe/torchfcpe/
transformers 4.38.2/einops/omegaconf/hydra/gradio 4.44.1/demucs/pedalboard/timm/smp/
torchmetrics/spafe/auraloss/gin/ml_collections/beartype ...)
ffmpeg 9.1（conda-forge，位于 runtime\Library\bin）
```

## 2. fairseq（关键风险点）

`fairseq 0.12.2` 以纯 Python 包方式内置，**无需 MSVC**：

```
>>> from fairseq import checkpoint_utils
>>> checkpoint_utils.load_model_ensemble_and_task(['pretrain/contentvec/checkpoint_best_legacy_500.pt'])
# 正常返回 models/saved_cfg/task，contentvec768l12tta2x 编码器可用
```

实测该路径完成了一次完整推理，说明 contentvec 加载链路真实可用（不是只 import 通过）。

## 3. 推理（CPU）

```
输入：4~6 秒 44.1kHz 单声道正弦测试音
模型：models/薇欧拉/model_12000.pt（RectifiedFlow, 1 spk）
参数：-d cpu -pe rmvpe -id 1，infer_step=50（rk4）
结果：RC=0，输出 517 KB wav，采样率 44.1kHz
耗时：≈ 2.1 s/step × 50 ≈ 105 s（2 核 CPU）
```

结论：rmvpe 音高提取 + contentvec 语义编码 + reflow 采样 + pc_nsf_hifigan 声码器
**全链路在 CPU 上跑通**。

## 4. WebUI

```
netstat: TCP 0.0.0.0:7860 LISTENING
GET /        -> 200, 328 KB
GET /config  -> 227 KB (gradio app graph 正常构建)
```

修复过的两个真实故障：

1. `gradio_client` 解析 JSON-Schema `additionalProperties: true`（bool）时抛
   `TypeError: argument of type 'bool' is not iterable` → `tools/patch_runtime.py` 幂等修复。
2. `fastapi/starlette` 版本过新导致 `TemplateResponse` 签名变化 →
   锁定 `fastapi==0.115.4` / `starlette==0.41.2`。

两者不修时表现为 launch 阶段报 `When localhost is not accessible, a shareable link must be created`，
很容易被误判成"网络/代理问题"。

## 5. 便携包（解压即用）验收

```
dist/DDSP-SVC-6.2-Portable-cu128.zip.001~003   4.05 GB
dist/DDSP-SVC-6.2-Weights.zip.001~002           3.25 GB
```

把 5 个分卷解压到 **全新目录** `C:\portest`（与构建目录无关）后：

```
runtime\python.exe setup\verify_env.py
  -> RESULT: PASS      （torch / fairseq / ffmpeg / 权重 全部 OK）
runtime\python.exe webui.py
  -> GET http://127.0.0.1:7860/ = 200
runtime\python.exe main_reflow.py ... -d cpu
  -> 正常输出 wav
```

即：`runtime` 目录是**可重定位**的（换路径不用重装），zip 内 `start.bat` 直接可用。

## 6. 未在本机验证的部分

- 真实 NVIDIA 显卡上的 CUDA kernel 执行（本机无卡）。代码路径为 `torch.cuda.is_available()` 分支，
  与 CPU 分支共用同一套模型代码；cu128 轮子已包含 sm_86/89/90/120 内核。
- MSST 音乐分离的完整跑通（需要更长音频与更多显存/内存；权重与 ffmpeg 已就位，模块 import 通过）。
