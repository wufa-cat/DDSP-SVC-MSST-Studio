# Create GitHub repo + push + release(assets) in one shot.
# Need: gh authed (gh auth login), and dist\ built (tools\make_assets.ps1)
# usage: powershell -ExecutionPolicy Bypass -File tools\create_repo.ps1 -Repo "owner/name"
param(
  [Parameter(Mandatory=$true)][string]$Repo,
  [string]$Tag = "v6.2.0",
  [string]$Description = "DDSP-SVC 6.2 + MSST WebUI Windows 开箱即用版：单套 torch 2.7.1+cu128 覆盖 RTX 30/40/50 系，无显卡回落 CPU，fairseq 纯 Python 内置免编译。WebUI 7860 / TensorBoard 6006",
  [switch]$Private,
  [switch]$SkipRelease
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

if (-not (Get-Command gh -EA SilentlyContinue)) { throw "gh not found: https://cli.github.com/" }
gh auth status | Out-Null

$topics = @("ddsp","ddsp-svc","voice-conversion","singing-voice-conversion","rectified-flow","reflow",
  "pytorch","cuda","windows","portable","fairseq","hubert","contentvec","rmvpe","hifigan",
  "music-source-separation","demucs","gradio","webui","audio")

Write-Host "[1/4] creating repo $Repo ..."
$vis = if ($Private) { "--private" } else { "--public" }
gh repo create $Repo $vis --description $Description --source . --remote origin --push

Write-Host "[2/4] setting topics ..."
foreach ($t in $topics) { gh repo edit $Repo --add-topic $t | Out-Null }

if ($SkipRelease) { Write-Host "skip release"; exit 0 }

$dist = Join-Path $Root "dist"
if (-not (Test-Path $dist)) { throw "dist\ not found - run tools\make_assets.ps1 first" }

Write-Host "[3/4] creating release $Tag and uploading assets ..."
$files = Get-ChildItem $dist -Filter "*.zip.00*" | ForEach-Object { $_.FullName }
$notes = Join-Path $Root ".github\release-notes\$Tag.md"
$args = @("release","create",$Tag) + $files + @("--title","$Tag · Windows 开箱即用（cu128 单环境 / 30·40·50 系通用）")
if (Test-Path $notes) { $args += @("--notes-file", $notes) } else { $args += @("--notes","see README") }
& gh @args

Write-Host "[4/4] fixing release.json urls + commit ..."
$man = Get-Content (Join-Path $dist "release.json") -Raw | ConvertFrom-Json
foreach ($a in $man.assets) { foreach ($p in $a.parts) {
  $p.url = "https://github.com/$Repo/releases/download/$Tag/" + $p.name } }
$man.repo = $Repo; $man.tag = $Tag
$man | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 "tools\release.json"
git add tools\release.json
git commit -m "chore(release): 指向 $Tag 附件"
git push
Write-Host "DONE -> https://github.com/$Repo/releases/tag/$Tag"
