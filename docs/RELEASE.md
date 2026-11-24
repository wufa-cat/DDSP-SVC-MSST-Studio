# 发布 Release 附件（维护者用）

## 1. 准备 7-Zip

把 `7za.exe` 放到 `tools\bin\7za.exe`（7-Zip 官网 → "7-Zip Extra" 包里的单文件版即可）。

## 2. 打包

```powershell
powershell -ExecutionPolicy Bypass -File tools\make_assets.ps1 `
  -Repo "yourname/DDSP-SVC-6.2-Win" -Tag "v6.2.0"
```

产出在 `dist\`：

| 文件 | 内容 |
|---|---|
| `DDSP-SVC-6.2-Portable-cu128.zip.00N` | runtime + 代码（解压即用） |
| `DDSP-SVC-6.2-Weights.zip.00N` | pretrain / models / MSST 权重 |

同时生成 `dist\release.json`（含每个分卷的 sha256 与字节数）。

## 3. 上传

在 GitHub 建 Release（tag 与 `-Tag` 一致），把 `dist\` 下所有分卷当附件上传。
单附件上限 2G，所以脚本按 1.9G 分卷。

## 4. 回填清单

把 `dist\release.json` 覆盖到 `tools\release.json`，提交。之后使用者执行：

```bat
runtime\python.exe tools\fetch_release.py --only all
```

即可自动下载分卷 → 校验 sha256 → 合并 → 解压。
