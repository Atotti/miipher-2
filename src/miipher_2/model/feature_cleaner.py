# ------------------------------------------------------------
#  miipher/model/feature_cleaner.py
# ------------------------------------------------------------
"""
Conformer 部分を廃止し、mHuBERT+ParallelAdapter へ置換した FeatureCleaner
"""

from torch import nn

from miipher_2.adapters.parallel_adapter import ParallelAdapter
from miipher_2.extractors.mhubert import MHubert9


class FeatureCleaner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = MHubert9()
        # mHuBERT の 12 Transformer ブロックに Adapter を挿入
        for blk in self.extractor.hubert.encoder.layers:
            blk.adapter = ParallelAdapter(dim=768, hidden=1024).to(blk.self_attn.q_proj.weight.device)
            original_forward = blk.forward

            def patched_forward(x, *args, _orig=original_forward, _ad=blk.adapter, **kwargs):
                return _ad(_orig(x, *args, **kwargs))

            blk.forward = patched_forward  # type: ignore

    def forward(self, wav16):
        """
        Returns: cleaned feature (B, 768, T*4) ← frame-rate は変えない
        """
        return self.extractor(wav16)
