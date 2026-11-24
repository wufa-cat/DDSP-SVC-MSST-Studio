# DDSP-SVC 6.2 × MSST — Windows 部署版

本套 WebUI 与整合工作流由作者制作并完成跨平台移植。
本分支（`-Win` 部署分支）补齐了 **Win 开箱即用环境**：不用手动装 Python、不用编译器、不挑显卡（30/40/50 系通吃）、无显卡自动回落 CPU。

## 内置测试模型
- `models/薇欧拉/`：由作者本人原创训练的测试模型，仅供部署验证与试听使用。

---

## 端口（两个，都记住）

| 端口 | 服务 | 何时用 |
|---|---|---|
| **7860** | WebUI 主界面 | 日常使用：推理、翻唱、分离、训练 |
| **6006** | TensorBoard | 训练时看 loss 曲线、Audio 页面试听 checkpoint |

- `start.bat` → 只起 WebUI（7860）
- `start_all.bat` → 同时起 TensorBoard（6006）+ WebUI（7860）（和原版 `start_windows.bat` 行为一致）
- `start_tensorboard.bat` → 单独起 TensorBoard

---

## 三种部署方式

### A. 便携包（推荐，解压即用，不联网）

1. 到 Releases 下载（GitHub 单附件上限 2G，所以分卷）：
   - `DDSP-SVC-6.2-Portable-cu128.zip.001~003` — 运行环境 + 代码，约 **4.05 G**
   - `DDSP-SVC-6.2-Weights.zip.001~002` — 权重，约 **3.25 G**
2. 合并分卷：`copy /b 分卷1+分卷2+... 目标.zip`，或用脚本自动搞定：
   ```bat
   python tools\fetch_release.py --only all
   ```
3. 解压到**纯英文路径**（如 `D:\DDSP-SVC-6.2`）。
4. 双击 `start.bat`。

### B. 仓库 + 一键装环境

```bat
git clone <仓库地址>
cd DDSP-SVC-6.2-Win
setup\install_env.bat
tools\fetch_release.py --only weights
start.bat
```

`install_env.bat` 自动：检测显卡 → 装 Miniconda 到 `%LOCALAPPDATA%`（免管理员）→ 建 `runtime\` → 装 torch 与全部依赖 → 装 ffmpeg → 解包内置 fairseq → 自检。

### C. 便携包 + 自建环境混搭

有便携包但想换环境：删掉 `runtime\`，跑 `setup\install_env.bat`，权重目录不动。

---

## 显卡适配

| 显卡 | 驱动 | torch | 说明 |
|---|---|---|---|
| RTX 50 系（sm_120） | ≥ 570 | 2.7.1+cu128 | 必须有 sm_120 内核，旧版跑不了 |
| RTX 40 系（sm_89） | ≥ 525 | 2.7.1+cu128 | 同一套环境 |
| RTX 30 系（sm_86） | ≥ 525 | 2.7.1+cu128 | 同一套环境 |
| 无独显 / 驱动太旧 | — | 自动 CPU | 能跑完，速度慢 |

一套 `cu128` 环境覆盖 30/40/50 系，这是选它而不是 cu121 的原因。
自检：`runtime\python.exe setup\verify_env.py`，会打印实际显卡、cc、显存。

---

## 关键设计说明

**fairseq 为什么内置？**
`fairseq==0.12.2` 在 Windows + Python 3.10 上要 MSVC 编译 Cython 扩展，几乎装不上。
本项目以**纯 Python 包**形式放入 `runtime\Lib\site-packages\fairseq`（源码包见 `setup\vendor\fairseq-0.12.2.tar.gz`），
contentvec 编码器用的 `load_model_ensemble_and_task` 可用，无需编译器。实测 `checkpoint_best_legacy_500.pt` 正常加载。

**ffmpeg 在哪？**
`runtime\Library\bin\ffmpeg.exe`（conda-forge）。`start.bat` 会把它加进 `PATH`——所以**必须用 start.bat 启动**，
否则 MSST 分离 / 音频转码会报 `ffmpeg not found`。

**权重为什么不在 git 里？**
约 3.9G（MSST 3.27G + DDSP pretrain 0.48G + 薇欧拉 0.28G），走 Release 附件 + `tools/fetch_release.py` 校验下载。

**中文路径？**
别用。torch 的 DLL 与 conda 环境对非 ASCII 路径偶发失败，统一纯英文。

## 目录

```
runtime\            便携 Python 3.10 + 全部依赖 + ffmpeg（不进 git）
setup\              安装器、依赖清单、内置 fairseq、自检脚本
tools\              fetch_release.py（拉附件）/ make_assets.ps1（打包发版）
pretrain\           底模（rmvpe / contentvec / nsf_hifigan）        ← 附件
models\             用户模型（薇欧拉）                              ← 附件
MSST\               音乐分离模块，权重在 MSST\pretrain\              ← 附件
exp\                训练实验目录（TensorBoard 读这里）
results\            推理输出
start.bat / start_all.bat / start_tensorboard.bat
```
