# DDSP-SVC 6.2 · Windows 整合版

DDSP-SVC 6.2 的歌声转换，加上 MSST 音乐分离和一套混音流程，打包成 Windows 上双击就能开的 WebUI。

原项目只给算法和命令行脚本；这个仓库把环境、ffmpeg、模型加载、五步翻唱流程、训练界面全部接好，解压即用。

---

## 能做什么

### 一条龙翻唱

拖一首歌进去，走完五步，出成品 wav：

1. **人声 / 伴奏分离** — MSST BS-RoFormer（Viperx 12.97）或 Mel-Band RoFormer（KJ）
2. **主唱 / 和声分离** — MSST Karaoke（Becruily），把主唱和原曲和声拆开
3. **主唱翻唱** — DDSP-SVC 推理，用你选的音色模型替换主唱
4. **和声处理** — 和声去混响后一起翻唱，或保留原曲和声直接叠
5. **混音** — 伴奏音量、轻量空间混响、输出格式，导出最终成品

产物按阶段落在 `results/` 下：`01_翻唱结果`、`02_一条龙翻唱`、`03_模型分离结果`、`04_混音与后期处理`。
每一步都有独立的追溯日志和历史记录，可以只重跑其中一步，也可以回放中间音轨对比。

### 分步使用

不想跑全流程就单独用某一步：

- **只翻唱** — 已有干声，直接上传 wav，选音色模型和参数出结果
- **只分离** — 出人声、伴奏、主唱、和声四条轨，单独下载
- **只混音** — 手上有翻唱干声和伴奏，自己调音量/混响出成品

### 训练自己的音色

界面里提供完整训练流程：上传素材（可自动切片、去静音、过滤过短片段）→ 提取特征 → 启动训练 →
TensorBoard（6006）看 loss 曲线并在 Audio 页面试听各步 checkpoint → 挑一个步数设为可用模型。

训练参数都暴露在界面上：批大小、学习率、衰减步数、验证间隔与比例、音高增强、缓存设备（cpu/cuda）、
是否缓存全部数据、训练方式。

### 模型管理

选择已有模型目录、上传 `.zip` 模型包（含 `.pt` 与 `config.yaml`）、从某一步权重继续微调、
下载精简模型包、删除模型；模型目录下的配置会自动跟随。

### 音频预处理

内置 FFmpeg 单文件处理：去低频底噪、统一音量、转 44.1kHz WAV，翻唱前把素材洗干净。

---

## 可调参数

| 参数 | 取值 | 说明 |
|---|---|---|
| 音高提取器 | rmvpe / fcpe / crepe | rmvpe 抗噪最好，fcpe 快，crepe 精度高但慢 |
| 采样方法 | auto / rk4 / euler | auto 跟随模型默认；rk4 音质细，euler 更快 |
| 采样步数 | 20–1000 | 默认 50；降到 20–30 明显更快，音质略降 |
| 变调 | 半音 | 整首歌升降 key |
| 共振峰偏移 | 半音 | 音高增强模型可用，改音色明暗而不改音高 |
| 声区偏移 | 半音 | 仅 pc_nsf_hifigan 声码器生效 |
| 静音阈值 / 间隔 | dB / ms | 训练素材切片用 |
| 声码器 | pc_nsf_hifigan（2025.02）/ nsf_hifigan | pc 版是现在推荐的 |

---

## 打开界面

```bat
start.bat              :: WebUI        -> http://127.0.0.1:7860
start_all.bat          :: WebUI + TensorBoard
start_tensorboard.bat  :: TensorBoard  -> http://127.0.0.1:6006
```

- **7860** — 主界面，翻唱/分离/混音/训练都在这里
- **6006** — TensorBoard，训练时看曲线、试听 checkpoint

必须用 `start.bat` 启动，它会把 `runtime\Library\bin`（ffmpeg）、各个缓存目录、`MSST_DIR` 一并配好。

---

## 装

### 便携包

Releases 里下载 5 个分卷（GitHub 单附件上限 2G，所以切开）：

- `DDSP-SVC-6.2-Portable-cu128.zip.001~003` — 运行环境 + 代码，4.05 G
- `DDSP-SVC-6.2-Weights.zip.001~002` — 权重，3.25 G

有 Python（或已解压过一次）时，一条命令搞定下载、校验和、合并、解压：

```bat
python tools\fetch_release.py --only all
```

什么都没有也能装：把 5 个分卷和 `tools\merge_volumes.bat` 一起下载到同一目录，双击 `merge_volumes.bat`
（纯 cmd 的 `copy /b`，不需要装 Python），再用 7-Zip / WinRAR 解压两个 zip 到当前目录。

解压到纯英文路径（`D:\DDSP-SVC-6.2` 这种），双击 `start.bat`。

### 从源码装

```bat
git clone <本仓库>
cd DDSP-SVC-6.2-Win
setup\install_env.bat                          :: 检测显卡 -> 建 runtime\ -> 装依赖 -> 装 ffmpeg -> 自检
runtime\python.exe tools\fetch_release.py --only weights
start.bat
```

`install_env.bat` 会顺带装 Miniconda（到 `%LOCALAPPDATA%`，不需要管理员），conda 走清华镜像。

---

## 显卡

| 显卡 | 驱动 | torch |
|---|---|---|
| RTX 50 系（sm_120） | ≥ 570 | 2.7.1+cu128 |
| RTX 40 系（sm_89） | ≥ 525 | 2.7.1+cu128 |
| RTX 30 系（sm_86） | ≥ 525 | 2.7.1+cu128 |
| 没有独显 | — | 自动用 CPU |

一套 cu128 环境同时带 sm_86 / sm_89 / sm_90 / sm_120 内核，所以 30/40/50 系不用分别装。
装的是 cu128 但驱动太旧时不会崩，直接走 CPU。

自检：

```bat
runtime\python.exe setup\verify_env.py
```

会打印 torch 版本、显卡型号与算力、依赖是否齐全、ffmpeg 是否可用、权重是否就位。

---

## 目录

```
runtime\      便携 Python 3.10 + 依赖 + ffmpeg（不进 git）
setup\        安装器、依赖清单、内置 fairseq、自检
tools\        fetch_release.py / make_assets.ps1 / patch_runtime.py / create_repo.*
pretrain\     rmvpe、contentvec、nsf_hifigan 底模          ← Release 附件
models\       音色模型（自带「薇欧拉」测试模型）           ← Release 附件
MSST\         分离模块，权重在 MSST\pretrain\              ← Release 附件
exp\          训练实验目录（TensorBoard 读这里）
results\      输出：01_翻唱结果 / 02_一条龙翻唱 / 03_模型分离结果 / 04_混音与后期处理
docs\         部署、发版、实测报告
```

---

## 这个整合版改了什么

- **fairseq 0.12.2 改成纯 Python 包内置**。原版 `pip install fairseq` 在 Windows + Python 3.10 上要 MSVC 编译
  Cython 扩展，绝大多数机器装不上；ContentVec 编码器需要的 `load_model_ensemble_and_task` 走纯 Python 版本完全够用。
- **一套 cu128 环境覆盖三代卡**，不用按显卡选轮子。
- **runtime 可重定位**，整个目录换个路径照样跑。
- **修了 gradio 的两个启动问题**：`gradio_client` 解析 `additionalProperties: true` 崩（`tools/patch_runtime.py` 幂等修复），
  以及 fastapi/starlette 版本过新导致的 `TemplateResponse` 签名不兼容（锁 `fastapi==0.115.4` / `starlette==0.41.2`，
  不锁的话会报 `localhost is not accessible`，很容易误判成网络问题）。
- **权重与代码分开分发**：权重 3.9 G 走 Release 附件带 sha256，仓库本体 11 MB。

实测数据（Windows Server 2022、2 核、8 G、无显卡、纯 CPU）：一次推理 4~6 秒音频 / 50 步约 **105 秒**，
输出 517 KB / 44.1 kHz wav；便携包换路径解压后自检 PASS、界面 200。详见 [docs/VALIDATION.md](docs/VALIDATION.md)。

---

## 常见问题

**启动报 `Cannot find empty port in range: 7860-7860`**
上次进程没退干净，`taskkill /f /im python.exe` 再来一次。

**报 `ffmpeg not found` 或分离失败**
没用 `start.bat` 启动，PATH 里没有 ffmpeg。

**50 系显卡报 CUDA 错**
驱动低于 570。

**能不能放中文路径**
不建议，torch 的 DLL 和便携 Python 对非 ASCII 路径偶发失败。

**显存/内存不够**
界面里可以调 batch、缓存设备、采样步数；步数降到 20~30 最省。

---

## 发版（维护者）

```powershell
powershell -ExecutionPolicy Bypass -File tools\make_assets.ps1 -Repo "<owner>/<repo>" -Tag "v6.2.0"
powershell -ExecutionPolicy Bypass -File tools\create_repo.ps1 -Repo "<owner>/<repo>"
```

代码变了只推代码；环境变了才重打 4 G 便携包；权重变了只重传 weights 分卷。详见 [docs/UPLOAD.md](docs/UPLOAD.md)。

## 许可与致谢

本仓库是部署整合，算法与模型来自上游：

- [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)（MIT, © yxlllc）— 歌声转换、RectifiedFlow、WebUI 原型
- [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)（MIT, © Roman Solovyev）— 音乐分离
- [ContentVec](https://github.com/auspicious3000/contentvec) / HuBERT — 语义编码器
- [RMVPE](https://github.com/Dream-High/RMVPE) — 音高提取
- OpenVPI `pc_nsf_hifigan` — 声码器
- 自带测试模型「薇欧拉」为本人训练，仅供部署验证与试听

各权重目录保留了上游的 `NOTICE.txt` / `STATEMENTS.txt`，一并遵守。本仓库新增脚本与文档同样为 MIT，见 [LICENSE](LICENSE)。
上游原始说明：[README_upstream.md](README_upstream.md) / [cn_README.md](cn_README.md)。
