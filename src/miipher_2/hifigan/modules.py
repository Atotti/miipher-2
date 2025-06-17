import torch
from torch import nn


class ResBlock1(nn.Module):
    def __init__(self, ch: int, k: int, d: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(ch, ch, k, 1, padding=(k * d - 1) // 2 * d, dilation=d),
                nn.Conv1d(ch, ch, k, 1, padding=(k * 1 - 1) // 2, dilation=1),
            ]
        )
        self.activation = nn.LeakyReLU(0.1, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c in self.convs:
            y = self.activation(x)
            y = c(y)
            x = y + x
        return x


def ResStack(channels: int) -> nn.Sequential:
    return nn.Sequential(
        ResBlock1(channels, 3, 1),
        ResBlock1(channels, 7, 1),
        ResBlock1(channels, 11, 1),
    )
