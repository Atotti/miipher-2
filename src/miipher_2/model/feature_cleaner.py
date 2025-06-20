from collections.abc import Callable
import functools
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

        # Adapterパッチ適用（DDP/FSDP対応版）
        for i, blk in enumerate(self.extractor.hubert.encoder.layers[:num_layers_to_patch]):
            adapter_module = self.adapters[i]

            # 動的に元のforward関数を取得する関数（バウンドメソッド問題を回避）
            def patched_forward(
                hidden_states: torch.Tensor,
                layer_idx: int = i,
                adapter_mod: ParallelAdapter = adapter_module,
            ) -> torch.Tensor:
                # 毎回動的に元のforwardを取得（DDP/FSDPで置き換わっても対応）
                original_ff = self.extractor.hubert.encoder.layers[layer_idx].feed_forward

                # 元のFeedForward(MLP)の出力を計算
                # original_ffはモジュールなので、__call__を使用
                ff_output = original_ff(hidden_states)

                # 同じ入力からAdapterの出力を計算
                adapter_output = adapter_mod(hidden_states)

                # 元のMLP出力にアダプターの出力を加算する
                return ff_output + adapter_output

            # feed_forwardモジュールのforwardメソッドを新しい関数で上書き
            blk.feed_forward.forward = functools.partial(patched_forward, layer_idx=i, adapter_mod=adapter_module)

            # LayerNormのパラメータは学習可能にする
            for param in blk.final_layer_norm.parameters():
                param.requires_grad = True

    def forward(self, wav16: torch.Tensor) -> torch.Tensor:
        return self.extractor(wav16)
