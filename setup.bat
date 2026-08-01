@echo off
setlocal
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
if errorlevel 1 (
  echo.
  echo Setup did not finish. Read the message above.
  pause
  exit /b 1
)
echo.
echo Setup complete. Run start.bat next.
pause
