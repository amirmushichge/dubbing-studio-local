@echo off
setlocal
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
if errorlevel 1 (
  echo.
  echo Установка не завершена. Прочитайте сообщение выше.
  pause
  exit /b 1
)
echo.
echo Готово. Теперь запустите start.bat
pause
