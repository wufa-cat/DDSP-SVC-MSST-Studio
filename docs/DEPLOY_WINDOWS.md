# 部署说明（Windows）

两套路径，任选其一。

---

## 路径 A：便携包（推荐，解压即用）

1. 打开 GitHub 仓库的 **Releases** 页面，下载同版本的：
   - `DDSP-SVC-6.2-Portable-cu128.zip.001`、`.002`、`.003`（运行环境 + 代码）
   - `DDSP-SVC-6.2-Weights.zip.001`、`.002`（模型权重）
2. 合并分卷（任意一种）：
   - 命令行：`copy /b DDSP-SVC-6.2-Portable-cu128.zip.001+...002+...003 DDSP-SVC-6.2-Portable-cu128.zip`
   - 或让脚本代劳：`python tools/fetch_release.py --only all`（自动下载、校验 sha256、合并、解压）
3. 解压到**纯英文路径**（例如 `D:\DDSP-SVC-6.2`，不要放桌面/中文目录）。
4. 双击 `start.bat` → 浏览器自动打开 `http://127.0.0.1:7860`。

便携包内已含：Python 3.10、torch 2.7.1+cu128、全部依赖、ffmpeg、纯 Python 版 fairseq。
**不需要装 Python、不需要装 conda、不需要联网。**

---

## 路径 B：仓库 + 一键装环境

```bat
git clone <你的仓库地址>
cd DDSP-SVC-6.2-Win
setup\install_env.bat          :: 自动检测显卡 → 建 runtime\ → 装依赖（约 10~20 分钟）
tools\fetch_release.py --only weights   :: 拉权重
start.bat
```

`install_env.bat` 做的事：检测 NVIDIA 显卡 → 下载 Miniconda（装到 `%LOCALAPPDATA%`，不需要管理员）→ `conda create -p runtime python=3.10` → 装 torch 与依赖 → 装 ffmpeg → 解包内置 fairseq → 跑 `setup\verify_env.py` 自检。

---

## 显卡适配表

| 显卡 | 驱动要求 | 用哪套 torch | 说明 |
|---|---|---|---|
| RTX 50 系（Blackwell, sm_120） | ≥ 570 | `torch 2.7.1+cu128` | 必须 cu128，旧版本没有 sm_120 内核 |
| RTX 40 系（sm_89） | ≥ 525（建议 ≥ 550） | `torch 2.7.1+cu128` | 同一套环境 |
| RTX 30 系（sm_86） | ≥ 525（建议 ≥ 550） | `torch 2.7.1+cu128` | 同一套环境 |
| 无独显 / 无驱动 | — | 自动回落 CPU | 能跑，但慢很多 |

判断方法：`nvidia-smi` 能正常输出 → 有驱动；`torch.cuda.is_available()` 为 `True` 才算真正可用。
装了 cu128 版 torch 但驱动太旧时，程序不会崩，会自动跑 CPU（`setup\verify_env.py` 会给出 warn）。

---

## 常见问题

**Q：为什么 fairseq 是"内置"的？**
A：`fairseq 0.12.2` 在 Windows + Python 3.10 上需要 MSVC 编译 Cython 扩展，绝大多数机器装不上。
本项目把 fairseq 作为**纯 Python 包**解包进 `runtime\Lib\site-packages\fairseq`（源码在 `setup\vendor\fairseq-0.12.2.tar.gz`），
contentvec 编码器需要的 `load_model_ensemble_and_task` 完全可用，无需编译器。

**Q：ffmpeg 在哪？**
A：在 `runtime\Library\bin\ffmpeg.exe`（conda-forge 版）。`start.bat` 会把它加进 `PATH`，所以必须通过 `start.bat` 启动，
否则 MSST 分离和音频格式转换会报 `ffmpeg not found`。

**Q：权重为什么不在 git 里？**
A：共约 3.9G（MSST 分离 3.27G + DDSP pretrain 0.48G + 薇欧拉 0.28G），git 仓库扛不住。走 Release 附件 + `tools/fetch_release.py`。

**Q：路径能带中文吗？**
A：不建议。conda 环境、torch 的 DLL 加载对非 ASCII 路径偶发失败，统一用纯英文路径。
