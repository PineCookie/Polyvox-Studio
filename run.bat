@echo off
rem UTF-8 mode is required for Torch Inductor on non-UTF-8 Windows locales.
set PYTHONUTF8=1
rem Models downloaded in the Models tab live here; settings left at their
rem defaults use them (see README: Models).
if not defined VOXCPM_STUDIO_MODELS_DIR set VOXCPM_STUDIO_MODELS_DIR=%~dp0models
if not defined VOXCPM_STUDIO_LORA_DIR set VOXCPM_STUDIO_LORA_DIR=%~dp0lora
if not defined VOXCPM_STUDIO_DATA_DIR set VOXCPM_STUDIO_DATA_DIR=%~dp0data
uv run --project "%~dp0." voxcpm-studio %*
