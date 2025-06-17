import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

# HiFi-GAN が想定するパラメータ
TARGET_SR = 22050
TARGET_HOP_SIZE = 256
TARGET_FRAME_RATE = TARGET_SR / TARGET_HOP_SIZE

# HuBERTのパラメータ
MHUBERT_SR = 16000
MHUBERT_STRIDE = 320
MHUBERT_FRAME_RATE = MHUBERT_SR / MHUBERT_STRIDE


class MHubertToMel(nn.Module):
    def __init__(self, hubert_dim: int) -> None:
        super().__init__()
        # 次元数をHuBERTの次元数から80に変換する
        self.proj = nn.Conv1d(hubert_dim, 80, 1)

        # フレームレートを変換するためのスケールファクター
        self.scale_factor = TARGET_FRAME_RATE / MHUBERT_FRAME_RATE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): HuBERTからの特徴量 (B, hubert_dim, T_hubert)
        Returns:
            torch.Tensor: HiFi-GAN用の特徴量 (B, 80, T_hifigan)
        """
        # 次元数を射影
        x = self.proj(x)

        # 時間軸を線形補間でリサンプリング
        return F.interpolate(x, scale_factor=self.scale_factor, mode="linear", align_corners=False)
