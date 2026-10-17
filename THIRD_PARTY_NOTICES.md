# Third-party notices

Polyvox Studio is licensed under the [Apache License 2.0](LICENSE). It includes
code from the projects below, under their own licenses; each license text is kept
next to the code. Changes made to vendored code are listed in each package's
`__init__.py`.

| Component | In this repository | From | License |
|---|---|---|---|
| VoxCPM training script | `src/polyvox/vendor/train_voxcpm_finetune.py` (unmodified) | [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) | Apache-2.0, Copyright OpenBMB ([text](src/polyvox/vendor/LICENSE-voxcpm)) |
| dots.tts | `src/polyvox/vendor/dots_tts/` | [rednote-hilab/dots.tts](https://github.com/rednote-hilab/dots.tts) 0.3.1 | Apache-2.0, Copyright dots.tts Team, RedNote ([text](src/polyvox/vendor/dots_tts/LICENSE)) |
| GPT-SoVITS | `src/polyvox/vendor/gpt_sovits/` | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | MIT, Copyright RVC-Boss ([text](src/polyvox/vendor/gpt_sovits/LICENSE)) |
| x-transformers (rotary embeddings) | `src/polyvox/vendor/gpt_sovits/f5_tts/rotary.py` | [lucidrains/x-transformers](https://github.com/lucidrains/x-transformers) 2.17.10 | MIT, Copyright Phil Wang ([text](src/polyvox/vendor/gpt_sovits/f5_tts/LICENSE-x-transformers)) |
| BS / Mel-Band RoFormer | `src/polyvox/vendor/roformer/` | [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) | MIT, Copyright Roman Solovyev (ZFTurbo) ([text](src/polyvox/vendor/roformer/LICENSE)) |

The GPT-SoVITS code includes parts its repository took from other projects (such
as the F5-TTS diffusion transformer, the ERes2Net speaker model from 3D-Speaker,
G2PW and the CMU Pronouncing Dictionary), as distributed in that repository.

Release archives also contain the built web UI, which bundles
[React](https://github.com/facebook/react) (MIT).

Python packages and models are not part of this repository: they are downloaded
when the studio is installed or used, under their own licenses. The Models tab
shows each model's license; some differ from the code's (the de-reverb model's
weights, for example, are GPL-3.0).

Fish Audio S2-Pro (`src/polyvox/engines/fish_s2/`) is this project's own
implementation of the published model architecture; it contains no Fish Audio
code. Its weights, downloaded in the Models tab, are under the Fish Audio
Research License: research and non-commercial use only, a commercial license
from Fish Audio otherwise.
