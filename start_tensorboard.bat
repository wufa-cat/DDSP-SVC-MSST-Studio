@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "ROOT=%CD%"
if not exist "%ROOT%\runtime\python.exe" ( echo [ERROR] runtime missing & pause & exit /b 1 )
set "PATH=%ROOT%\runtime;%ROOT%\runtime\Scripts;%ROOT%\runtime\Library\bin;%PATH%"
set "PYTHONUTF8=1"
"%ROOT%\runtime\python.exe" -m tensorboard.main --logdir "%ROOT%\exp" --host 127.0.0.1 --port 6006
pause
