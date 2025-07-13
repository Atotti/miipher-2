from collections.abc import Callable

import torch
from omegaconf import DictConfig
from torch import nn

from miipher_2.adapters.parallel_adapter import ParallelAdapter
from miipher_2.extractors.hubert import SSLExtractor


class FeatureCleaner(nn.Module):
    def __init__(self, cfg_model: DictConfig) -> None:
        super().__init__()
        model_type = cfg_model.get("model_type", "auto")
        self.extractor = SSLExtractor(
            model_name=cfg_model.hubert_model_name,
            layer=cfg_model.hubert_layer - 1,
            model_type=model_type,
        )

        # ベースとなるSSLモデルの全パラメータを凍結
        self.extractor.model.eval()
        for param in self.extractor.model.parameters():
            param.requires_grad = False

        hubert_dim = self.extractor.model.config.hidden_size

        num_layers_to_patch = cfg_model.hubert_layer

        self.adapters = nn.ModuleList(
            [ParallelAdapter(dim=hubert_dim, hidden=cfg_model.adapter_hidden_dim) for _ in range(num_layers_to_patch)]
        )

        for i, blk in enumerate(self.extractor.model.encoder.layers[:num_layers_to_patch]):
            original_ff_forward = blk.feed_forward.forward
            adapter_module = self.adapters[i]

            # Adapterを挿入
            def patched_forward(
                hidden_states: torch.Tensor,
                _orig_ff: Callable = original_ff_forward,
                _ad: ParallelAdapter = adapter_module,
            ) -> torch.Tensor:
                # 元のFeedForward(MLP)の出力を計算
                ff_output = _orig_ff(hidden_states)

                # 同じ入力からAdapterの出力を計算
                adapter_output = _ad(hidden_states)

                # 元のMLP出力にアダプターの出力を加算する
                return ff_output + adapter_output

            # feed_forwardモジュールのforwardメソッドを新しい関数で上書き
            blk.feed_forward.forward = patched_forward

            for param in blk.final_layer_norm.parameters():
                param.requires_grad = True

    def forward(self, wav16: torch.Tensor) -> torch.Tensor:
        return self.extractor(wav16)
