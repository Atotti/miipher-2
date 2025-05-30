from dataclasses import dataclass

import torch
from peft import PeftConfig, PeftType
from torch import nn


@dataclass
class ParallelAdapterConfig(PeftConfig):
    """Minimal PEFT config used only to store hyper-parameters."""

    peft_type: str | PeftType | None = "PARALLEL_ADAPTER"
    reduction_factor: int = 8  # hidden // reduction
    non_linearity: str = "gelu"  # "gelu" or "relu"


class ParallelAdapter(nn.Module):
    """Bottleneck MLP whose output is **added** to the FFN output (parallel)."""

    def __init__(self, hidden_size: int, cfg: ParallelAdapterConfig) -> None:
        super().__init__()
        bottleneck = hidden_size // cfg.reduction_factor
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.nonlinear: nn.Module
        if cfg.non_linearity.lower() == "relu":
            self.nonlinear = nn.ReLU()
        else:
            self.nonlinear = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        return self.up(self.nonlinear(self.down(x)))
