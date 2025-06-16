import torch
from torch import nn


class ParallelAdapter(nn.Module):
    def __init__(self, dim: int, hidden: int = 1024) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        # Xavier 初期化を 0.01 スケールで
        for m in self.ff:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        return x + self.ff(x)
