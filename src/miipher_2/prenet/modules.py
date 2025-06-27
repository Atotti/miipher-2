import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10_000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (T, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(-1)].T  # (B, C, T)


class ConformerLayer(nn.Module):
    """HuBERT-compatible lightweight Conformer."""

    def __init__(self, dim: int, heads: int = 8, ff_mult: float = 1.0) -> None:
        super().__init__()
        ff_dim = int(dim * ff_mult)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_dim * 4),
            nn.SiLU(),
            nn.Linear(ff_dim * 4, dim),
            nn.Dropout(0.1),
        )
        self.mha = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.conv = ConvModule(dim)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_dim * 4),
            nn.SiLU(),
            nn.Linear(ff_dim * 4, dim),
            nn.Dropout(0.1),
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        x = x + 0.5 * self.ffn1(x)
        attn, _ = self.mha(x, x, x)
        x = x + self.drop(attn)
        x = x.transpose(1, 2)  # (B, C, T)
        x = x + self.conv(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)


class ConvModule(nn.Module):
    """Depth-wise Conv1D + GLU (ESPnet style)."""

    def __init__(self, channels: int, kernel_size: int = 31) -> None:
        super().__init__()
        self.pointwise = nn.Conv1d(channels, 2 * channels, 1)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=channels,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        x = self.pointwise(x)
        x = F.glu(x, dim=1)
        x = self.depthwise(x)
        x = self.norm(x)
        return self.proj(self.act(x))
