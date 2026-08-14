@echo off
chcp 65001 >nul
title LAAP Brain API (Aris)
echo ============================================
echo  LAAP Brain API  -  Aris
echo  http://localhost:11546
echo  关闭本窗口即停止服务
echo ============================================
echo.
G:\laap\.venv\Scripts\python.exe G:\laap\launcher.py
pause
