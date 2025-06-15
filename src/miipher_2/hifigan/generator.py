from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from miipher_2.hifigan.modules import ResStack
from miipher_2.hifigan.prenet import MHubertToMel


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int = 768,  # ← 実際には Prenet側で 128ch に変換
        channels: int = 512,
        out_channels: int = 1,
        kernel_size: int = 7,
        upsample_rates: Sequence[int] = (5, 4, 2, 2),
        upsample_kernel_sizes: Sequence[int] = (10, 8, 4, 4),
        resblock_kernel_sizes: Sequence[int] = (3, 7, 11),
        resblock_dilations: Sequence[Sequence[int]] = ((1, 3, 5),) * 3,
    ) -> None:
        super().__init__()

        # --- Prenet: 768→128 ch, 50 Hz→200 Hz -----------------
        self.prenet = MHubertToMel()  # (B,128,T50) -> (B,128,T200)

        # --- Initial conv -------------------------------------
        self.pre_conv = nn.Conv1d(128, channels, kernel_size, 1, padding=(kernel_size - 1) // 2)

        # --- Upsample layers ----------------------------------
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=False)):
            self.ups.append(
                nn.ConvTranspose1d(
                    channels // (2**i),
                    channels // (2 ** (i + 1)),
                    k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )
            # ResStack after each upsample
            for _ in range(3):  # three ResBlocks per stage
                self.resblocks.append(ResStack(channels // (2 ** (i + 1))))

        # --- Final layers -------------------------------------
        self.post_conv = nn.Sequential(
            nn.LeakyReLU(0.1, True),
            nn.Conv1d(
                channels // (2 ** len(upsample_rates)),
                out_channels,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.Tanh(),
        )

        self.apply(self._weight_init)

    # ----------------------------------------------------------
    @staticmethod
    def _weight_init(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight, 0.0, 0.02)

    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 768, T50)  — mHuBERT features
        Returns:
            wav : (B, 1, T_wave)  — 16 kHz waveform
        """
        x = self.prenet(x)  # (B,128,T200)
        x = self.pre_conv(x)  # (B,512,T200)

        resblock_idx = 0
        for up in self.ups:
            x = F.leaky_relu(x, 0.1)
            x = up(x)  # upsample

            # three ResBlocks each stage
            xs: list[torch.Tensor] = []
            for _ in range(3):
                xs.append(self.resblocks[resblock_idx](x))
                resblock_idx += 1
            # average outputs
            x = sum(xs) / len(xs)

        return self.post_conv(x)
