import torch
from torch import nn
import torch.nn.functional as F

# HiFi-GAN UNIVERSAL_V1が想定するパラメータ
TARGET_SR = 22050
TARGET_HOP_SIZE = 256
TARGET_FRAME_RATE = TARGET_SR / TARGET_HOP_SIZE # 86.1328... Hz

# mHuBERTのパラメータ
MHUBERT_SR = 16000
MHUBERT_STRIDE = 320
MHUBERT_FRAME_RATE = MHUBERT_SR / MHUBERT_STRIDE # 50.0 Hz

class MHubertToMel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 1. 次元数を768から80に変換する
        self.proj = nn.Conv1d(768, 80, 1)

        # 2. フレームレートを変換するためのスケールファクター
        self.scale_factor = TARGET_FRAME_RATE / MHUBERT_FRAME_RATE # 約1.7226

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): mHuBERTからの特徴量 (B, 768, T_hubert)
        Returns:
            torch.Tensor: HiFi-GAN用の特徴量 (B, 80, T_hifigan)
        """
        # 次元数を射影
        x = self.proj(x)

        # 時間軸を線形補間でリサンプリング
        # (B, 80, T_hubert) -> (B, 80, T_hifigan)
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode='linear',
            align_corners=False
        )
        return x
