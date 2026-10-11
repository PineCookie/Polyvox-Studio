import sys
import os
import torch

from polyvox.vendor.gpt_sovits.eres2net.ERes2NetV2 import ERes2NetV2
from polyvox.vendor.gpt_sovits.eres2net import kaldi as Kaldi


class SV:
    def __init__(self, device, is_half, sv_path):
        pretrained_state = torch.load(sv_path, map_location="cpu", weights_only=True)
        embedding_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        embedding_model.load_state_dict(pretrained_state)
        embedding_model.eval()
        self.embedding_model = embedding_model
        if is_half == False:
            self.embedding_model = self.embedding_model.to(device)
        else:
            self.embedding_model = self.embedding_model.half().to(device)
        self.is_half = is_half

    def compute_embedding3(self, wav):
        with torch.no_grad():
            if self.is_half == True:
                wav = wav.half()
            feat = torch.stack(
                [Kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for wav0 in wav]
            )
            sv_emb = self.embedding_model.forward3(feat)
        return sv_emb
