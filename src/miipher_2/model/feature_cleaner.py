from collections.abc import Callable
from typing import Any

import torch
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

        # ベースとなるHuBERTの全パラメータを凍結
        self.extractor.hubert.eval()
        for param in self.extractor.hubert.parameters():
            param.requires_grad = False

        hubert_dim = self.extractor.hubert.config.hidden_size

        num_layers_to_patch = cfg_model.hubert_layer + 1

        self.adapters = nn.ModuleList(
            [ParallelAdapter(dim=hubert_dim, hidden=cfg_model.adapter_hidden_dim) for _ in range(num_layers_to_patch)]
        )

        # 指定されたレイヤーまでループして、forwardをパッチする
        for i, blk in enumerate(self.extractor.hubert.encoder.layers[:num_layers_to_patch]):
            original_forward = blk.forward
            adapter_module = self.adapters[i]

            def patched_forward(
                x: torch.Tensor,
                *args: Any,  # noqa: ANN401
                _orig: Callable[..., tuple[torch.Tensor, ...]] = original_forward,
                _ad: ParallelAdapter = adapter_module,
                **kwargs: Any,  # noqa: ANN401
            ) -> tuple[torch.Tensor, ...]:
                original_outputs = _orig(x, *args, **kwargs)
                hidden_states = original_outputs[0]
                modified_hidden_states = _ad(hidden_states)
                return (modified_hidden_states,) + original_outputs[1:]

            blk.forward = patched_forward

    def forward(self, wav16: torch.Tensor) -> torch.Tensor:
        return self.extractor(wav16)
