@echo off
chcp 65001 >nul
cd /d %~dp0
set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
set PYTHONUNBUFFERED=1
set PYTHONUTF8=1
set MSST_DIR=%~dp0MSST
start "TensorBoard-6006" /min python -m tensorboard.main --logdir=exp --host=0.0.0.0 --port=6006
python webui.py
pause
