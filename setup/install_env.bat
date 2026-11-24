@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0.."
set "ROOT=%CD%"
set "LOG=%ROOT%\setup\install.log"
set "PY=%ROOT%\runtime\python.exe"
set "TUNA=https://pypi.tuna.tsinghua.edu.cn/simple"
if not exist "%ROOT%\setup" mkdir "%ROOT%\setup"
break > "%LOG%"

echo ============================================================
echo  DDSP-SVC 6.2  -  one-click environment setup
echo  install dir : %ROOT%\runtime
echo  log         : %LOG%
echo ============================================================

rem ---------- 1. detect GPU ----------
set "TARGET=cpu"
set "GPUNAME=none"
where nvidia-smi >nul 2>&1
if errorlevel 1 goto :nogpu
for /f "usebackq delims=" %%i in (`nvidia-smi --query-gpu^=name^,driver_version --format^=csv^,noheader 2^>nul`) do if not defined GPUNAME set "GPUNAME=%%i"
if "%GPUNAME%"=="none" goto :nogpu
if "%GPUNAME%"=="" goto :nogpu
set "TARGET=cu128"
echo [1/7] GPU detected : %GPUNAME%
goto :detected
:nogpu
echo [1/7] no NVIDIA GPU / driver - CPU mode build

:detected
echo        torch build  : %TARGET%

rem ---------- 2. bootstrap conda (only if runtime missing) ----------
if exist "%PY%" (
  echo [2/7] runtime already exists, skip conda bootstrap
  goto :installdeps
)
set "MC=%LOCALAPPDATA%\DDSP-Miniconda3"
if exist "%MC%\python.exe" goto :mkvenv
echo [2/7] downloading Miniconda3 (py3.10) ...
curl.exe -L --fail -o "%TEMP%\ddsp_mc3.exe" https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_24.9.2-0-Windows-x86_64.exe >> "%LOG%" 2>&1
if errorlevel 1 (
  echo       TUNA mirror failed, retry official repo ...
  curl.exe -L --fail -o "%TEMP%\ddsp_mc3.exe" https://repo.anaconda.com/miniconda/Miniconda3-py310_24.9.2-0-Windows-x86_64.exe >> "%LOG%" 2>&1
)
if not exist "%TEMP%\ddsp_mc3.exe" ( echo [FAIL] cannot download Miniconda & goto :fail )
echo       installing Miniconda to %MC% ...
start /wait "" "%TEMP%\ddsp_mc3.exe" /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=%MC%
if not exist "%MC%\python.exe" ( echo [FAIL] Miniconda install failed & goto :fail )

:mkvenv
echo [2/7] creating python 3.10 env at runtime\ ...
rem china mirror first (repo.anaconda.com is often unreachable), fall back to defaults
call "%MC%\condabin\conda.bat" create -p "%ROOT%\runtime" -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r --override-channels python=3.10 -y >> "%LOG%" 2>&1
if not exist "%PY%" (
  echo       mirror failed, retry with default channels ...
  call "%MC%\condabin\conda.bat" create -p "%ROOT%\runtime" python=3.10 -y >> "%LOG%" 2>&1
)
if not exist "%PY%" ( echo [FAIL] conda create failed, see %LOG% & goto :fail )

:installdeps
echo [3/7] upgrading pip ...
"%PY%" -m pip install --upgrade pip >> "%LOG%" 2>&1

echo [4/7] installing torch (%TARGET%) ...
if /i "%TARGET%"=="cu128" (
  "%PY%" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 -r "%ROOT%\setup\requirements\torch_cu128.txt" >> "%LOG%" 2>&1
) else (
  "%PY%" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r "%ROOT%\setup\requirements\torch_cpu.txt" >> "%LOG%" 2>&1
)
if errorlevel 1 ( echo [FAIL] torch install failed, see %LOG% & goto :fail )

echo [5/7] installing runtime dependencies (this takes a few minutes) ...
"%PY%" -m pip install --no-cache-dir -i %TUNA% -r "%ROOT%\setup\requirements\base.txt" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo       TUNA failed, retry with default index ...
  "%PY%" -m pip install --no-cache-dir -r "%ROOT%\setup\requirements\base.txt" >> "%LOG%" 2>&1
)
"%PY%" -m pip install --no-cache-dir -i %TUNA% --no-deps asteroid==0.7.0 >> "%LOG%" 2>&1
"%PY%" -m pip install --no-cache-dir -i %TUNA% numpy==1.26.4 >> "%LOG%" 2>&1

echo [6/7] installing ffmpeg (conda-forge) ...
call "%MC%\Scripts\conda.exe" install -p "%ROOT%\runtime" -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge ffmpeg -y --override-channels >> "%LOG%" 2>&1

echo [7/7] installing bundled fairseq (pure python, no MSVC needed) ...
if exist "%ROOT%\setup\vendor\fairseq-0.12.2.tar.gz" (
  if exist "%TEMP%\fairseq-0.12.2" rmdir /s /q "%TEMP%\fairseq-0.12.2"
  tar -xzf "%ROOT%\setup\vendor\fairseq-0.12.2.tar.gz" -C "%TEMP%"
  if exist "%ROOT%\runtime\Lib\site-packages\fairseq" rmdir /s /q "%ROOT%\runtime\Lib\site-packages\fairseq"
  xcopy /E /I /Y /Q "%TEMP%\fairseq-0.12.2\fairseq" "%ROOT%\runtime\Lib\site-packages\fairseq" >nul
  echo       fairseq unpacked
) else (
  echo [WARN] setup\vendor\fairseq-0.12.2.tar.gz missing - contentvec encoder will not load
)

echo [7b/7] applying runtime compatibility patches ...
"%PY%" "%ROOT%\tools\patch_runtime.py" >> "%LOG%" 2>&1

echo.
echo ============================================================
echo  verifying environment ...
"%PY%" "%ROOT%\setup\verify_env.py"
if errorlevel 1 goto :fail
echo ============================================================
echo  DONE. double-click start.bat to launch the WebUI.
echo ============================================================
pause
exit /b 0

:fail
echo.
echo ============================================================
echo  SETUP FAILED - see %LOG%
echo ============================================================
pause
exit /b 1
