from torch import nn


class MHubertToMel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(768, 384, 4, 2, 1),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose1d(384, 128, 4, 2, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv1d(128, 128, 1),
        )
        # He 正規分布初期化
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.main(x)  # (B, 128, 4*T) => 200 Hz
