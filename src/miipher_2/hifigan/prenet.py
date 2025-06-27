import math

import torch
import torch.nn.functional as F  # noqa: N812
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


class Miipher2PreNet(nn.Module):
    """
    *prenet* used in Miipher-2, extended to emit **mel80 @ 86 fps** directly.

    Parameters
    ----------
    in_dim : int
        Hidden size of USM/HuBERT features (e.g. 768 for mHuBERT-147,
        1024 for japanese-hubert-large, 1536 for USM-2B).
    n_layers : int, default 4
        Number of Conformer blocks (paper uses 4).
    mel_dim : int, default 80
        Output channels expected by HiFi-GAN.
    src_fps : Literal[50.0, ...], default 50.0
        Frame-rate of the encoder features (16 kHz / 320 stride).
    tgt_hop : int, default 256
        Hop length of the downstream vocoder (22 050 Hz / 256 = 86.1328 fps).
    """

    def __init__(
        self,
        *,
        in_dim: int,
        n_layers: int = 4,
        mel_dim: int = 80,
        src_fps: float = 50.0,
        tgt_hop: int = 256,
        sr: int = 22_050,
    ) -> None:
        super().__init__()
        self.pos_enc = PositionalEncoding(in_dim)
        self.layers = nn.ModuleList([ConformerLayer(in_dim) for _ in range(n_layers)])

        # ---------- mel80 projection ----------
        self.proj = nn.Conv1d(in_dim, mel_dim, kernel_size=1)

        # scale factor for linear interpolation
        tgt_fps = sr / tgt_hop
        self.register_buffer("scale", torch.tensor(tgt_fps / src_fps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, in_dim, T_enc)
            USM/HuBERT encoder features at ~50 fps.

        Returns
        -------
        mel : Tensor, shape (B, 80, T_vocoder)
            Mel-like features aligned with HiFi-GAN hop size.
        """
        x = self.pos_enc(x)  # (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)  # back to (B, C, T)

        x = self.proj(x)  # (B, 80, T_enc)
        return F.interpolate(
            x,
            scale_factor=float(self.scale),
            mode="linear",
            align_corners=False,
        )
