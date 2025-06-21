import functools
from collections.abc import Callable
from types import MethodType
from typing import Any, Optional

import torch
from omegaconf import DictConfig
from torch import nn

from miipher_2.adapters.parallel_adapter import ParallelAdapter
from miipher_2.extractors.hubert import HubertExtractor


class FeatureCleaner(nn.Module):
    """
    HuBERTの特徴量にAdapterを適用して清浄化するモジュール
    メモリ効率向上のため、外部からHubertExtractorを注入可能
    """

    def __init__(self, cfg_model: DictConfig, hubert_extractor: HubertExtractor | None = None) -> None:
        super().__init__()

        # 外部から提供されなければ、自分で作成する
        if hubert_extractor is None:
            self.extractor = HubertExtractor(
                model_name=cfg_model.hubert_model_name,
                layer=cfg_model.hubert_layer,
            )
        else:
            # 外部から注入されたインスタンスを使用（メモリ効率）
            self.extractor = hubert_extractor

        self.cfg_model = cfg_model

        # Adapterを初期化
        self._initialize_adapters()

        # HuBERTの特定レイヤーにAdapterを挿入
        self._patch_hubert_layers()

    def _initialize_adapters(self) -> None:
        """Adapterモジュールを初期化"""
        hubert_config = self.extractor.hubert.config

        # パッチ対象レイヤー数を制限（計算量とVRAM節約）
        num_layers_to_patch = self.cfg_model.get("hubert_layer", hubert_config.num_hidden_layers) + 1
        num_layers_to_patch = min(num_layers_to_patch, hubert_config.num_hidden_layers)

        self.adapters = nn.ModuleList()

        for _ in range(num_layers_to_patch):
            adapter = ParallelAdapter(
                dim=hubert_config.hidden_size,
                hidden=self.cfg_model.get("adapter_hidden_dim", 1024),  # 正しい引数名を使用
            )
            self.adapters.append(adapter)

    def _patch_hubert_layers(self) -> None:
        """
        HuBERTレイヤーのFFNにAdapterを適用（正しいforward methodパッチ）
        """
        # パッチ対象レイヤー数を制限
        num_layers_to_patch = len(self.adapters)

        for layer_idx, (layer, adapter) in enumerate(
            zip(self.extractor.hubert.encoder.layer[:num_layers_to_patch], self.adapters, strict=True)
        ):
            # ★★★ 正しいforward methodを保存（chunk_feed_forwardは存在しない）
            original_forward = layer.feed_forward.forward

            # functools.partialを使用してbound method問題を回避
            # これによりDDP/FSDPでも安全に動作する
            patched_forward = functools.partial(
                self._adapter_forward,
                original_forward=original_forward,
                adapter=adapter,
                layer_idx=layer_idx,
            )

            # ★★★ torch.nn.ModuleにMethodTypeでバインド
            layer.feed_forward.forward = MethodType(patched_forward, layer.feed_forward)

    def _adapter_forward(
        self,
        attention_output: torch.Tensor,
        *,
        original_forward: Callable[[torch.Tensor], torch.Tensor],
        adapter: ParallelAdapter,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Adapterを含むFFNのforward実装（正しいシグネチャ）
        """
        # 元のFFN処理
        ff_output = original_forward(attention_output)

        # Adapter処理（並列結合）
        adapter_output = adapter(attention_output)

        # 残差接続
        return ff_output + adapter_output

    def forward(self, wav16: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav16: 16kHzの音声波形 (B, T)
        Returns:
            特徴量 (B, hidden_size, T')
        """
        return self.extractor(wav16)
