from torch import nn

from miipher_2.adapters.parallel_adapter import ParallelAdapter
from miipher_2.extractors.hubert import MHubert9


class FeatureCleaner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = MHubert9()

        # 1. まず、ベースとなるHuBERTの全パラメータを凍結する
        self.extractor.hubert.eval()
        for param in self.extractor.hubert.parameters():
            param.requires_grad = False

        # 2. その後、学習させたいAdapterをアタッチする
        for blk in self.extractor.hubert.encoder.layers:
            blk.adapter = ParallelAdapter(dim=768, hidden=1024)

            original_forward = blk.forward

            def patched_forward(x, *args, _orig=original_forward, _ad=blk.adapter, **kwargs):
                original_outputs = _orig(x, *args, **kwargs)
                hidden_states = original_outputs[0]
                modified_hidden_states = _ad(hidden_states)
                return (modified_hidden_states,) + original_outputs[1:]

            blk.forward = patched_forward

    def forward(self, wav16):
        return self.extractor(wav16)
