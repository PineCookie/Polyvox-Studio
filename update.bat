@echo off
rem Updates this folder to the latest Polyvox Studio release. Models, datasets (data), voices
rem (lora) and the Python environment stay; close the studio first. The work is done by
rem tools\update.ps1, run from a temporary copy so that it can replace this folder's files.
setlocal
set "UPDATER=%TEMP%\polyvox-update-%RANDOM%%RANDOM%.ps1"
copy /y "%~dp0tools\update.ps1" "%UPDATER%" >nul || (echo tools\update.ps1 is missing. & pause & exit /b 1)
rem One block: cmd reads it whole before the update replaces this file.
(powershell -NoProfile -ExecutionPolicy Bypass -File "%UPDATER%" -Folder "%~dp0." %* & del "%UPDATER%" >nul 2>nul & pause & exit /b)
