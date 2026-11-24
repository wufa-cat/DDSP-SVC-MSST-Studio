# Build portable + weights release assets (volume-split zip) and emit release.json
# usage:
#   powershell -ExecutionPolicy Bypass -File tools\make_assets.ps1 -Repo owner/repo -Tag v6.2.0
param(
  [string]$Repo = "OWNER/DDSP-SVC-6.2-Win",
  [string]$Tag  = "v6.2.0",
  [int]$VolumeMB = 1900,
  [string]$OutDir = "dist",
  [switch]$SkipPortable,
  [switch]$SkipWeights
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$SevenZip = Join-Path $Root "tools\bin\7za.exe"
if (-not (Test-Path $SevenZip)) { throw "missing tools\bin\7za.exe (7-Zip Extra: 7za.exe)" }
if (-not (Test-Path (Join-Path $Root "runtime\python.exe"))) { throw "runtime\ not found - build env first (setup\install_env.bat)" }

$Dist = Join-Path $Root $OutDir
$Stage = Join-Path $Dist "_stage"
New-Item -ItemType Directory -Force -Path $Dist | Out-Null
if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null

function Trim-Runtime([string]$rt) {
  $kill = @(
    "Lib\site-packages\torch\include", "Lib\site-packages\torch\test",
    "Lib\site-packages\torch\lib\*.pdb", "Lib\site-packages\torch\lib\*.lib",
    "Lib\site-packages\**\__pycache__", "Lib\site-packages\**\tests",
    "Lib\site-packages\**\*.pyc", "Library\lib\*.lib", "Library\lib\*.a",
    "Library\include", "Lib\site-packages\pip\_vendor\distlib\t64.exe",
    "conda-meta", "pkgs", "share\jupyter", "Scripts\*.pdb"
  )
  foreach ($p in $kill) {
    Get-ChildItem -Path (Join-Path $rt $p) -Force -ErrorAction SilentlyContinue |
      Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
  }
}

if (-not $SkipPortable) {
  Write-Host "[1/3] staging portable tree ..."
  $app = Join-Path $Stage "app"
  New-Item -ItemType Directory -Force -Path $app | Out-Null
  $excludeDirs = @("runtime","dist","downloads","cache","exp","raw","results","downloads",".git",".github")
  # code + configs (weights excluded - they live in the weights asset)
  robocopy $Root $app /E /NFL /NDL /NJH /NJS /NP `
    /XD (Join-Path $Root "runtime") (Join-Path $Root $OutDir) (Join-Path $Root ".git") `
        (Join-Path $Root "MSST\pretrain") (Join-Path $Root "models") `
        (Join-Path $Root "exp") (Join-Path $Root "raw") (Join-Path $Root "results") `
    /XF *.log *.pyc | Out-Null
  New-Item -ItemType Directory -Force -Path (Join-Path $app "MSST\pretrain") | Out-Null
  New-Item -ItemType Directory -Force -Path (Join-Path $app "models")         | Out-Null
  New-Item -ItemType Directory -Force -Path (Join-Path $app "exp")            | Out-Null
  New-Item -ItemType Directory -Force -Path (Join-Path $app "cache")          | Out-Null

  Write-Host "[2/3] staging runtime (trimmed) ..."
  robocopy (Join-Path $Root "runtime") (Join-Path $app "runtime") /E /NFL /NDL /NJH /NJS /NP `
    /XD (Join-Path $Root "runtime\pkgs") | Out-Null
  Trim-Runtime (Join-Path $app "runtime")
}

if (-not $SkipWeights) {
  Write-Host "[3/3] staging weights ..."
  $w = Join-Path $Stage "weights"
  New-Item -ItemType Directory -Force -Path $w | Out-Null
  foreach ($d in @("pretrain","models","MSST\pretrain")) {
    $src = Join-Path $Root $d
    if (Test-Path $src) {
      $dst = Join-Path $w $d
      New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
      robocopy $src $dst /E /NFL /NDL /NJH /NJS /NP | Out-Null
    }
  }
}

function New-VolumeZip([string]$srcDir, [string]$zipName) {
  $target = Join-Path $Dist $zipName
  Get-ChildItem -Path $Dist -Filter "$zipName.*" -ErrorAction SilentlyContinue | Remove-Item -Force
  if (Test-Path $target) { Remove-Item $target -Force }
  Push-Location $srcDir
  & $SevenZip a -tzip -mx=5 "-v${VolumeMB}m" $target * | Out-Null
  Pop-Location
  Get-ChildItem -Path $Dist -Filter "$zipName.*" | Sort-Object Name
}

$assets = @()
function Add-Asset([string]$group, [string[]]$parts, [string]$logicalName) {
  $entries = @()
  $total = 0
  foreach ($p in $parts) {
    $h = (Get-FileHash -Path $p -Algorithm SHA256).Hash.ToLower()
    $len = (Get-Item $p).Length
    $total += $len
    $entries += [pscustomobject]@{
      name   = (Split-Path $p -Leaf)
      size   = $len
      sha256 = $h
      url    = "https://github.com/$Repo/releases/download/$Tag/" + (Split-Path $p -Leaf)
    }
  }
  return [pscustomobject]@{
    group   = $group
    name    = $logicalName
    size    = $total
    extract = $true
    parts   = $entries
  }
}

if (-not $SkipPortable) {
  $parts = New-VolumeZip (Join-Path $Stage "app") "DDSP-SVC-6.2-Portable-cu128.zip"
  $assets += Add-Asset "runtime" $parts.FullName "DDSP-SVC-6.2-Portable-cu128.zip"
}
if (-not $SkipWeights) {
  $parts = New-VolumeZip (Join-Path $Stage "weights") "DDSP-SVC-6.2-Weights.zip"
  $assets += Add-Asset "weights" $parts.FullName "DDSP-SVC-6.2-Weights.zip"
}

$manifest = [pscustomobject]@{ repo = $Repo; tag = $Tag; assets = $assets }
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $Dist "release.json")

Remove-Item $Stage -Recurse -Force
Write-Host ""
Write-Host "done. assets in $Dist"
$manifest.assets | ForEach-Object { "{0,-10} {1,-45} {2,8:N0} MB" -f $_.group, $_.name, ($_.size / 1MB) }
Write-Host "manifest: $Dist\release.json  (copy to tools\release.json after upload)"
