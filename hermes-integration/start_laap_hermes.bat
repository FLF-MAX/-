@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: LAAP + Hermes Integration Launcher
:: Usage: start_laap_hermes.bat [port]

set LAAP_PORT=%1
if "%LAAP_PORT%"=="" set LAAP_PORT=11546

:: Auto-detect LAAP root from this batch file's location.
:: Override by setting LAAP_ROOT before running this script.
if "%LAAP_ROOT%"=="" set LAAP_ROOT=%~dp0..
set LAAP_API_BASE=http://localhost:%LAAP_PORT%
set ARIS_BRAIN=%LAAP_ROOT%\aris_brain
if "%HERMES_HOME%"=="" set HERMES_HOME=%LOCALAPPDATA%\hermes\hermes-agent

echo ============================================================
echo  LAAP Brain + Hermes Agent Integration
echo ============================================================
echo LAAP API:     %LAAP_API_BASE%
echo LAAP root:    %LAAP_ROOT%
echo Hermes home:  %HERMES_HOME%
echo.

:: Kill any existing LAAP API on this port
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%LAAP_PORT%" ^| findstr "LISTENING"') do (
    echo Stopping existing LAAP API on port %LAAP_PORT% (PID %%a)
    taskkill /PID %%a /F >nul 2>&1
)

:: Start LAAP API in background
echo [1/3] Starting LAAP Brain API on port %LAAP_PORT%...
start "LAAP Brain API" /MIN cmd /c "cd /d %ARIS_BRAIN% && python laap_brain_api.py --port %LAAP_PORT%"

:: Wait for API to be ready
echo [2/3] Waiting for LAAP API to be ready...
:wait_loop
timeout /t 1 /nobreak >nul
curl -s -o nul -w "%%{http_code}" %LAAP_API_BASE%/health > %TEMP%\laap_health.tmp 2>nul
set /p HEALTH=<%TEMP%\laap_health.tmp
if not "!HEALTH!"=="200" goto wait_loop
echo LAAP API is ready (HTTP %HEALTH%).

:: Launch Hermes MCP server in stdio mode via Hermes config
echo [3/3] Hermes MCP integration configured.
echo.
echo To start chatting with LAAP modulation:
echo   hermes chat --skills laap-bridge
echo.
echo To start Hermes gateway:
echo   hermes gateway
echo.
echo Press any key to stop LAAP API and exit...
pause >nul

:: Cleanup
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%LAAP_PORT%" ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)
echo LAAP API stopped.
