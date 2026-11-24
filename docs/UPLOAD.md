# 发布到自己的 Git 仓库（维护者操作顺序）

## 0. 产物在哪

在构建机上：`C:\DDSP-SVC-6.2-Win\dist\`

```
DDSP-SVC-6.2-Portable-cu128.zip.001   1.90 G
DDSP-SVC-6.2-Portable-cu128.zip.002   1.90 G
DDSP-SVC-6.2-Portable-cu128.zip.003   0.25 G
DDSP-SVC-6.2-Weights.zip.001          1.90 G
DDSP-SVC-6.2-Weights.zip.002          1.35 G
release.json                          附件清单（含每个分卷的 sha256）
```

## 1. 推代码

```bat
:: 方式一：已有仓库
git init -b main
git add -A
git commit -m "DDSP-SVC 6.2 Win 部署版"
git remote add origin https://github.com/<你>/<仓库>.git
git push -u origin main

:: 方式二：从 bundle 还原（本包附带 ddsp-svc-6.2-win.bundle）
git clone ddsp-svc-6.2-win.bundle DDSP-SVC-6.2-Win
cd DDSP-SVC-6.2-Win
git remote set-url origin https://github.com/<你>/<仓库>.git
git push -u origin main
```

> 提交前确认 `.gitignore` 生效：`git status` 里**不应**出现 `runtime/`、`dist/`、`pretrain/*.pt`、`models/`。

## 2. 建 Release 并上传 5 个分卷

网页方式：仓库 → Releases → Draft a new release → tag `v6.2.0` → 把 `dist\` 里 5 个分卷拖进附件区 → Publish。

命令行方式（GitHub CLI）：

```bat
gh release create v6.2.0 ^
  dist\DDSP-SVC-6.2-Portable-cu128.zip.001 ^
  dist\DDSP-SVC-6.2-Portable-cu128.zip.002 ^
  dist\DDSP-SVC-6.2-Portable-cu128.zip.003 ^
  dist\DDSP-SVC-6.2-Weights.zip.001 ^
  dist\DDSP-SVC-6.2-Weights.zip.002 ^
  --title "DDSP-SVC 6.2 Win 部署版" ^
  --notes "torch 2.7.1+cu128 单环境覆盖 RTX 30/40/50 系；无显卡自动 CPU"
```

## 3. 回填 release.json

`dist\release.json` 里的 URL 是按打包时的 `-Repo` 参数生成的占位地址。上传完成后：

- 确认里面 `https://github.com/<owner>/<repo>/releases/download/v6.2.0/...` 与你实际仓库一致；
- 覆盖到 `tools\release.json` 并提交：

```bat
copy /y dist\release.json tools\release.json
git add tools\release.json
git commit -m "chore: 指向 v6.2.0 release 附件"
git push
```

如果不一致，重新生成一次：

```powershell
powershell -ExecutionPolicy Bypass -File tools\make_assets.ps1 -Repo "<你>/<仓库>" -Tag "v6.2.0"
```

## 4. 别人怎么用

```bat
git clone https://github.com/<你>/<仓库>.git
cd <仓库>
runtime\python.exe tools\fetch_release.py --only all   :: 没有 runtime 时先用系统 python3 跑
start.bat
```

或者纯手动：下载 5 个分卷 → `copy /b 分卷... 目标.zip` → 解压 → `start.bat`。

## 5. 版本升级怎么发

改了代码但环境没变：只发代码（用户 `git pull` 即可），**不需要**重传 4G 便携包。
环境变了（pip 依赖/ffmpeg/torch 版本）：重跑 `tools\make_assets.ps1`，tag 递增（`v6.2.1`）。
权重变了：只重传 weights 分卷。
