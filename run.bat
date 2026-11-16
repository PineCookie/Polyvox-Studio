@echo off
setlocal
rem UTF-8 mode is required for Torch Inductor on non-UTF-8 Windows locales.
set PYTHONUTF8=1
rem Models downloaded in the Models tab live here; settings left at their
rem defaults use them (see docs/GUIDE.md: Models).
if not defined POLYVOX_MODELS_DIR set POLYVOX_MODELS_DIR=%~dp0models
if not defined POLYVOX_LORA_DIR set POLYVOX_LORA_DIR=%~dp0lora
if not defined POLYVOX_DATA_DIR set POLYVOX_DATA_DIR=%~dp0data

rem uv installs Python and the dependencies; offer to install it when it's missing.
where uv >nul 2>nul
if not errorlevel 1 goto have_uv
echo uv is not installed. It installs Python and everything else the studio needs
echo (https://docs.astral.sh/uv/, a single program).
set "ANSWER=y"
set /p "ANSWER=Install uv now? [Y/n] "
if /i "%ANSWER:~0,1%"=="n" goto no_uv
rem Started from PowerShell 7, its module path would make Windows PowerShell load the wrong modules.
set "PSModulePath="
powershell -NoProfile -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
if errorlevel 1 goto no_uv
rem The installer puts uv on PATH for new windows; this one needs it too.
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
if defined XDG_BIN_HOME set "PATH=%XDG_BIN_HOME%;%PATH%"
if defined UV_INSTALL_DIR set "PATH=%UV_INSTALL_DIR%;%PATH%"
where uv >nul 2>nul
if not errorlevel 1 goto have_uv
echo uv was installed, but this window can't find it yet: open a new one and run this again.
goto fail

:no_uv
echo Install uv, then run this again:
echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
goto fail

:have_uv

rem Release zips include the built web UI; a git checkout builds it once (Node.js 20+).
if exist "%~dp0web\dist\index.html" goto run
where npm >nul 2>nul
if not errorlevel 1 goto build
echo The web UI is not built. Download a release zip, which includes it,
echo or install Node.js 20+ and run this again.
goto fail

:build
echo Building the web UI, first run only...
pushd "%~dp0web"
call npm ci || goto build_failed
call npm run build || goto build_failed
popd

:run
rem The PyTorch build for this machine: cu130, cu126 or cpu (POLYVOX_TORCH overrides).
set "TORCH_VARIANT="
for /f "delims=" %%v in ('uv run --no-project --python 3.13 "%~dp0src\polyvox\torch_variant.py"') do set "TORCH_VARIANT=%%v"
if not defined TORCH_VARIANT goto fail
echo PyTorch build: %TORCH_VARIANT%
rem uv installs Python and the dependencies on the first run (and swaps PyTorch builds).
uv run --project "%~dp0." --no-default-groups --group %TORCH_VARIANT% polyvox %*
if errorlevel 1 goto fail
exit /b 0

:build_failed
popd
echo Building the web UI failed.

:fail
rem Opened by a double-click, the window would close before the message can be read.
pause
exit /b 1
