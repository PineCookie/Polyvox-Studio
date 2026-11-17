@echo off
setlocal
rem UTF-8 mode is required for Torch Inductor on non-UTF-8 Windows locales.
set PYTHONUTF8=1
rem Models downloaded in the Models tab live here; settings left at their
rem defaults use them (see docs/GUIDE.md: Models).
if not defined POLYVOX_MODELS_DIR set POLYVOX_MODELS_DIR=%~dp0models
if not defined POLYVOX_LORA_DIR set POLYVOX_LORA_DIR=%~dp0lora
if not defined POLYVOX_DATA_DIR set POLYVOX_DATA_DIR=%~dp0data
rem Started from PowerShell 7, its module path would make Windows PowerShell load the wrong modules.
set "PSModulePath="

rem uv installs Python and the dependencies; offer to install it when it's missing.
where uv >nul 2>nul
if not errorlevel 1 goto have_uv
echo uv is not installed. It installs Python and everything else the studio needs
echo (https://docs.astral.sh/uv/, a single program).
set "ANSWER=y"
set /p "ANSWER=Install uv now? [Y/n] "
if /i "%ANSWER:~0,1%"=="n" goto no_uv
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
rem Mirrors in mainland China: asked once before the first install, the answer kept in .mirror
rem (cn or off; POLYVOX_MIRROR overrides it).
if not defined POLYVOX_MIRROR if exist "%~dp0.mirror" set /p POLYVOX_MIRROR=<"%~dp0.mirror"
if not defined POLYVOX_MIRROR if not exist "%~dp0.venv\" call :ask_mirror
set "FROZEN="
if /i "%POLYVOX_MIRROR%"=="cn" (call :use_mirror) else (call :lock_urls official)
if errorlevel 1 goto fail

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
uv run %FROZEN% --project "%~dp0." --no-default-groups --group %TORCH_VARIANT% polyvox %*
if errorlevel 1 goto fail
exit /b 0

:build_failed
popd
echo Building the web UI failed.

:fail
rem Opened by a double-click, the window would close before the message can be read.
pause
exit /b 1

:ask_mirror
rem Suggested when Windows is set to China's time zone or a zh-CN locale.
set "SUGGEST=n"
for /f "delims=" %%z in ('tzutil /g 2^>nul') do if /i "%%z"=="China Standard Time" set "SUGGEST=y"
for /f "tokens=3" %%l in ('reg query "HKCU\Control Panel\International" /v LocaleName 2^>nul') do if /i "%%l"=="zh-CN" set "SUGGEST=y"
set "CHOICES=[y/N]"
if "%SUGGEST%"=="y" set "CHOICES=[Y/n]"
echo.
echo The first start downloads Python, PyTorch and other packages (several GB). In mainland
echo China, mirrors are usually much faster: Tsinghua University for packages, Nanjing
echo University for PyTorch and Python, hf-mirror.com for models.
set "ANSWER=%SUGGEST%"
set /p "ANSWER=Download from these mirrors? %CHOICES% "
set "POLYVOX_MIRROR=off"
if /i "%ANSWER:~0,1%"=="y" set "POLYVOX_MIRROR=cn"
> "%~dp0.mirror" echo %POLYVOX_MIRROR%
echo Saved in .mirror (cn or off): change or delete it to choose again.
exit /b 0

:use_mirror
echo Downloading from mirrors in mainland China (.mirror).
if not defined UV_DEFAULT_INDEX set "UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple"
if not defined UV_PYTHON_INSTALL_MIRROR set "UV_PYTHON_INSTALL_MIRROR=https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone"
rem The Models tab's default download source (one chosen there still wins).
if not defined POLYVOX_HF_ENDPOINT set "POLYVOX_HF_ENDPOINT=https://hf-mirror.com"
rem The lock's file addresses point at the mirrors; --frozen installs exactly what it lists
rem (versions and hashes), without resolving again against other indexes.
set "FROZEN=--frozen"
call :lock_urls mirror
exit /b

:lock_urls
rem Rewrites uv.lock's download addresses: "mirror" (the same files on the mirrors) or
rem "official". Only when it holds the other kind.
set "FROM=pypi.tuna.tsinghua.edu.cn/packages/ mirrors.nju.edu.cn/pytorch/whl/"
if "%1"=="mirror" set "FROM=files.pythonhosted.org/packages/ download-r2.pytorch.org/whl/cu download-r2.pytorch.org/whl/cpu/"
if not exist "%~dp0uv.lock" exit /b 0
findstr /l "%FROM%" "%~dp0uv.lock" >nul || exit /b 0
powershell -NoProfile -ExecutionPolicy ByPass -File "%~dp0tools\lock_urls.ps1" -Lock "%~dp0uv.lock" -To %1
exit /b
