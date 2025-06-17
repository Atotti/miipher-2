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

        # 1. ベースとなるHuBERTの全パラメータを凍結
        self.extractor.hubert.eval()
        for param in self.extractor.hubert.parameters():
            param.requires_grad = False

        hubert_dim = self.extractor.hubert.config.hidden_size

        # ===== ここから修正 =====
        # 2. 指定されたレイヤーの数だけAdapterを作成する
        #    +1しているのは、0-basedのレイヤーインデックスを数に変換するため
        num_layers_to_patch = cfg_model.hubert_layer + 1

        self.adapters = nn.ModuleList(
            [ParallelAdapter(dim=hubert_dim, hidden=cfg_model.adapter_hidden_dim) for _ in range(num_layers_to_patch)]
        )

        # 3. 指定されたレイヤーまでループして、forwardをパッチする
        for i, blk in enumerate(self.extractor.hubert.encoder.layers[:num_layers_to_patch]):
            original_forward = blk.forward
            adapter_module = self.adapters[i]

            def patched_forward(x, *args, _orig=original_forward, _ad=adapter_module, **kwargs):
                original_outputs = _orig(x, *args, **kwargs)
                hidden_states = original_outputs[0]
                modified_hidden_states = _ad(hidden_states)
                return (modified_hidden_states,) + original_outputs[1:]

            blk.forward = patched_forward

    def forward(self, wav16):
        return self.extractor(wav16)
