@echo off
rem 兼容旧入口：原版 start_windows.bat 依赖已激活的 conda 环境
rem 这里转发到 start.bat，自动配置 runtime\、ffmpeg、PATH 与各缓存目录
call "%~dp0start.bat"
