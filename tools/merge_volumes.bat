@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0\.."
echo 合并 Release 分卷（纯 cmd，无需装任何东西）
echo.
set "PAIRS=DDSP-SVC-6.2-Portable-cu128.zip DDSP-SVC-6.2-Weights.zip"
for %%Z in (%PAIRS%) do (
  if exist "%%Z.001" (
    echo   合并 %%Z
    set "LIST="
    for /l %%i in (1,1,99) do (
      set "N=00%%i"
      set "N=!N:~-3!"
      if exist "%%Z.!N!" set "LIST=!LIST!+%%Z.!N!"
    )
    set "LIST=!LIST:~1!"
    cmd /c copy /b "!LIST!" "%%Z" >nul && echo     -^> %%Z  OK
  )
)
echo.
echo 完成。接下来用 7-Zip / WinRAR 解压这两个 zip 到本目录（覆盖合并）。
pause
