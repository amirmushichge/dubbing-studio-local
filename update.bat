@echo off
setlocal
chcp 65001 >nul
git -C "%~dp0" pull --ff-only
if errorlevel 1 pause & exit /b 1
call "%~dp0setup.bat"
