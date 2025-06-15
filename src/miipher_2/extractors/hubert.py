# ------------------------------------------------------------
#  miipher/extractors/mhubert.py
# ------------------------------------------------------------
"""
Frozen mHuBERT (147‑lang, base, 2nd‑iter) 抽出器。
9層目 (index=9) の hidden_states を (B,768,T) で返します。
"""

from typing import List, Union

import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

_MODEL_NAME = "utter-project/mHuBERT-147-base-2nd-iter"


class MHubert9(torch.nn.Module):
    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        super().__init__()
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.hubert = HubertModel.from_pretrained(model_name, output_hidden_states=True).eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: (B, T) float32, 16 kHz, ‑1 〜 1
        Returns:
            feat: (B, 768, T/320) 50 Hz
        """
        hs: list[torch.Tensor] = self.hubert(wav, output_hidden_states=True, return_dict=True).hidden_states
        # 9番目 (0‑based) を転置
        return hs[9].transpose(1, 2).contiguous()
