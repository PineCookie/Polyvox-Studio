@echo off
rem UTF-8 mode is required for Torch Inductor on non-UTF-8 Windows locales.
set PYTHONUTF8=1
if not defined VOXCPM_STUDIO_MODEL set VOXCPM_STUDIO_MODEL=%~dp0models\openbmb__VoxCPM2
if not defined VOXCPM_STUDIO_LORA_DIR set VOXCPM_STUDIO_LORA_DIR=%~dp0lora
if not defined VOXCPM_STUDIO_DATA_DIR set VOXCPM_STUDIO_DATA_DIR=%~dp0data
if not defined VOXCPM_STUDIO_ASR_MODEL if exist "%~dp0models\Qwen__Qwen3-ASR-1.7B-hf" set VOXCPM_STUDIO_ASR_MODEL=%~dp0models\Qwen__Qwen3-ASR-1.7B-hf
uv run --project "%~dp0." voxcpm-studio %*
