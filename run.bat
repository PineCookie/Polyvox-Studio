@echo off
rem UTF-8 mode is required for Torch Inductor on non-UTF-8 Windows locales.
set PYTHONUTF8=1
if not defined VOXCPM_STUDIO_MODEL set VOXCPM_STUDIO_MODEL=%~dp0models\openbmb__VoxCPM2
if not defined VOXCPM_STUDIO_LORA_DIR set VOXCPM_STUDIO_LORA_DIR=%~dp0lora
if not defined VOXCPM_STUDIO_DATA_DIR set VOXCPM_STUDIO_DATA_DIR=%~dp0data
if not defined VOXCPM_STUDIO_ASR_MODEL if exist "%~dp0models\Qwen__Qwen3-ASR-1.7B-hf" set VOXCPM_STUDIO_ASR_MODEL=%~dp0models\Qwen__Qwen3-ASR-1.7B-hf
if not defined VOXCPM_STUDIO_ALIGN_MODEL if exist "%~dp0models\Qwen__Qwen3-ForcedAligner-0.6B-hf" set VOXCPM_STUDIO_ALIGN_MODEL=%~dp0models\Qwen__Qwen3-ForcedAligner-0.6B-hf
if not defined VOXCPM_STUDIO_CTC_ALIGN_MODEL if exist "%~dp0models\jonatasgrosman__wav2vec2-large-xlsr-53-chinese-zh-cn" set VOXCPM_STUDIO_CTC_ALIGN_MODEL=%~dp0models\jonatasgrosman__wav2vec2-large-xlsr-53-chinese-zh-cn
rem Optional clean-up models for dataset import (vocal separation, de-reverb, denoise).
set "M=%~dp0models"
if not defined VOXCPM_STUDIO_VOCALS_MODEL if exist "%M%\pcunwa__Kim-Mel-Band-Roformer-FT\kimmel_unwa_ft2_bleedless.ckpt" set "VOXCPM_STUDIO_VOCALS_MODEL=%M%\pcunwa__Kim-Mel-Band-Roformer-FT\kimmel_unwa_ft2_bleedless.ckpt"
if not defined VOXCPM_STUDIO_DEREVERB_MODEL if exist "%M%\anvuew__dereverb_mel_band_roformer\dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt" set "VOXCPM_STUDIO_DEREVERB_MODEL=%M%\anvuew__dereverb_mel_band_roformer\dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"
if not defined VOXCPM_STUDIO_DENOISE_MODEL if exist "%M%\aufr33__melband-roformer-denoise\denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt" set "VOXCPM_STUDIO_DENOISE_MODEL=%M%\aufr33__melband-roformer-denoise\denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"
uv run --project "%~dp0." voxcpm-studio %*
