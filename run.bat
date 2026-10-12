@echo off
setlocal
rem UTF-8 mode is required for Torch Inductor on non-UTF-8 Windows locales.
set PYTHONUTF8=1
rem Models downloaded in the Models tab live here; settings left at their
rem defaults use them (see docs/GUIDE.md: Models).
if not defined POLYVOX_MODELS_DIR set POLYVOX_MODELS_DIR=%~dp0models
if not defined POLYVOX_LORA_DIR set POLYVOX_LORA_DIR=%~dp0lora
if not defined POLYVOX_DATA_DIR set POLYVOX_DATA_DIR=%~dp0data

where uv >nul 2>nul
if errorlevel 1 (
  echo uv is not installed. Install it, then run this again:
  echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  exit /b 1
)

rem Release zips include the built web UI; a git checkout builds it once (Node.js 20+).
if exist "%~dp0web\dist\index.html" goto run
where npm >nul 2>nul
if errorlevel 1 (
  echo The web UI is not built. Download a release zip, which includes it,
  echo or install Node.js 20+ and run this again.
  exit /b 1
)
echo Building the web UI, first run only...
pushd "%~dp0web"
call npm ci || goto build_failed
call npm run build || goto build_failed
popd

:run
rem The PyTorch build for this machine: cu130, cu126 or cpu (POLYVOX_TORCH overrides).
set "TORCH_VARIANT="
for /f "delims=" %%v in ('uv run --no-project --python 3.13 "%~dp0src\polyvox\torch_variant.py"') do set "TORCH_VARIANT=%%v"
if not defined TORCH_VARIANT exit /b 1
echo PyTorch build: %TORCH_VARIANT%
rem uv installs Python and the dependencies on the first run (and swaps PyTorch builds).
uv run --project "%~dp0." --no-default-groups --group %TORCH_VARIANT% polyvox %*
exit /b

:build_failed
popd
echo Building the web UI failed.
exit /b 1