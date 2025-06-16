from omegaconf import DictConfig
from torch import nn

from miipher_2.adapters.parallel_adapter import ParallelAdapter
from miipher_2.extractors.hubert import HubertExtractor


class FeatureCleaner(nn.Module):
    def __init__(self, cfg_model: DictConfig) -> None:
        super().__init__()
        self.extractor = HubertExtractor(
            model_name=cfg_model.hubert_model_name,
            layer=cfg_model.hubert_layer,
        )

        # 1. まず、ベースとなるHuBERTの全パラメータを凍結する
        self.extractor.hubert.eval()
        for param in self.extractor.hubert.parameters():
            param.requires_grad = False

        # 2. その後、学習させたいAdapterをアタッチする
        hubert_dim = self.extractor.hubert.config.hidden_size

        for blk in self.extractor.hubert.encoder.layers:
            blk.adapter = ParallelAdapter(dim=hubert_dim, hidden=cfg_model.adapter_hidden_dim)
            original_forward = blk.forward

            def patched_forward(x, *args, _orig=original_forward, _ad=blk.adapter, **kwargs):
                original_outputs = _orig(x, *args, **kwargs)
                hidden_states = original_outputs[0]
                modified_hidden_states = _ad(hidden_states)
                return (modified_hidden_states,) + original_outputs[1:]

            blk.forward = patched_forward

    def forward(self, wav16):
        return self.extractor(wav16)
