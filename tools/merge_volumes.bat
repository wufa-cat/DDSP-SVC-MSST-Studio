@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
set "BASE=%~dp0"
if exist "%BASE%..\DDSP-SVC-6.2-Portable-cu128.zip.001" if not exist "%BASE%DDSP-SVC-6.2-Portable-cu128.zip.001" set "BASE=%~dp0..\"
if not exist "%BASE%DDSP-SVC-6.2-Portable-cu128.zip.001" if not exist "%BASE%DDSP-SVC-6.2-Weights.zip.001" (
  echo [x] 没找到分卷 .001，请把本 bat 与 .001/.002/.003 放同一文件夹再双击
  pause & exit /b 1
)
cd /d "%BASE%"
for %%Z in (DDSP-SVC-6.2-Portable-cu128.zip DDSP-SVC-6.2-Weights.zip) do (
  if exist "%%Z.001" (
    if exist "%%Z" (
      echo   跳过 %%Z（已存在）
    ) else (
      set "LIST="
      for /l %%i in (1,1,99) do (
        set "N=00%%i"
        set "N=!N:~-3!"
        if exist "%%Z.!N!" set "LIST=!LIST!+%%Z.!N!"
      )
      set "LIST=!LIST:~1!"
      echo   合并 %%Z
      cmd /c copy /b "!LIST!" "%%Z" >nul && echo     -^> OK
    )
  )
)
echo.
echo 完成。现在用 7-Zip / WinRAR 解压这两个 zip 到本目录。
pause
