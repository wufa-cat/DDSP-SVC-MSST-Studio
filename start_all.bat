@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "ROOT=%CD%"
if not exist "%ROOT%\runtime\python.exe" ( echo [ERROR] runtime missing - run setup\install_env.bat & pause & exit /b 1 )
set "PATH=%ROOT%\runtime;%ROOT%\runtime\Scripts;%ROOT%\runtime\Library\bin;%PATH%"
set "MSST_DIR=%ROOT%\MSST"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"
set "HF_HOME=%ROOT%\cache\hf"
set "TORCH_HOME=%ROOT%\cache\torch"
set "NUMBA_CACHE_DIR=%ROOT%\cache\numba"
set "MPLCONFIGDIR=%ROOT%\cache\mpl"
set "GRADIO_ANALYTICS_ENABLED=False"
if not exist "%ROOT%\cache" mkdir "%ROOT%\cache"

echo  [1/2] TensorBoard on http://127.0.0.1:6006  (training curves / audio preview)
start "TensorBoard-6006" /min "%ROOT%\runtime\python.exe" -m tensorboard.main --logdir "%ROOT%\exp" --host 127.0.0.1 --port 6006
echo  [2/2] WebUI        on http://127.0.0.1:7860
start "" cmd /c "timeout /t 12 >nul & start "" http://127.0.0.1:7860"
"%ROOT%\runtime\python.exe" "%ROOT%\webui.py"
echo.
echo webui exited with code %ERRORLEVEL%
pause
