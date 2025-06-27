import torch
from torch import nn

from miipher_2.prenet.modules import ConformerLayer, PositionalEncoding


class ConformerPrenet(nn.Module):
    """768-ch 入力をそのまま 768-ch で出力する Prenet"""

    def __init__(self, in_dim: int = 768, n_layers: int = 4) -> None:
        super().__init__()
        self.pos_enc = PositionalEncoding(in_dim)
        self.layers = nn.ModuleList([ConformerLayer(in_dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)  → Conformer は (B, T, C) が欲しい
        x = self.pos_enc(x)  # PE は (B, C, T) でそのまま使える
        x = x.transpose(1, 2)  # (B, T, C)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)  # (B, C, T) に戻す
