import torch.nn.functional as F
from torch import nn


class MPDSub(nn.Module):
    def __init__(self, period) -> None:
        super().__init__()
        self.period = period
        chan = [32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList()
        for ch in chan:
            self.convs.append(nn.Conv2d(1, ch, (5, 1), (3, 1), padding=(2, 0)))
        self.conv_post = nn.Conv2d(chan[-1], 1, (3, 1), 1, padding=(1, 0))
        self.act = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        b, c, t = x.shape
        if t % self.period:  # pad so that t % period == 0
            pad_len = self.period - (t % self.period)
            x = F.pad(x, (0, pad_len), "reflect")
            t = t + pad_len
        x = x.view(b, 1, t // self.period, self.period)  # (B,1,T/P,P)
        fmap = []
        for c in self.convs:
            x = self.act(c(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11, 13, 17, 19)) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([MPDSub(p) for p in periods])

    def forward(self, x):
        rets = []
        for d in self.discriminators:
            rets.append(d(x))
        return rets  # list[(score, fmap)]


class LayerDiscriminator(nn.Module):
    def __init__(self, ch_mult) -> None:
        super().__init__()
        ch = [1, 32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList()
        for i in range(len(ch) - 1):
            self.convs.append(nn.Conv1d(ch[i], ch[i + 1], 5, 2, 2))
        self.conv_post = nn.Conv1d(ch[-1], 1, 3, 1, 1)
        self.act = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        fmap = []
        for c in self.convs:
            x = self.act(c(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([LayerDiscriminator(i) for i in range(3)])

    def forward(self, x):
        rets = []
        for i, d in enumerate(self.discriminators):
            if i in {1, 2}:
                x = F.avg_pool1d(x, 4, 2, padding=1)
            rets.append(d(x))
        return rets
