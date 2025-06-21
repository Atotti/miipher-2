import math

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

        # より正確な時間軸変換のための制御
        self.use_precise_resampling = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): HuBERTからの特徴量 (B, hubert_dim, T_hubert)
        Returns:
            torch.Tensor: HiFi-GAN用の特徴量 (B, 80, T_hifigan)
        """
        # 次元数を射影
        x = self.proj(x)

        if self.use_precise_resampling:
            # より正確な時間軸リサンプリング（long sequences対応）
            # exact target lengthを計算してsize指定でリサンプリング
            input_length = x.size(-1)
            target_length = math.ceil(input_length * self.scale_factor)
            return F.interpolate(x, size=target_length, mode="linear", align_corners=False)
        # 従来の scale_factor 方式（non-integer ratioで drift する可能性）
        return F.interpolate(x, scale_factor=self.scale_factor, mode="linear", align_corners=False)


class SincResampler(nn.Module):
    """
    バンド制限付きSinc補間によるリサンプリング（実験用）
    長尺音声での位相ドリフトを最小化
    """

    def __init__(self, input_sr: float, output_sr: float, lowpass_filter_width: int = 6) -> None:
        super().__init__()
        self.resampling_ratio = output_sr / input_sr
        self.lowpass_filter_width = lowpass_filter_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sinc補間による高品質リサンプリング
        """
        # PyTorchの実装では複雑になるため、現在は線形補間のfallback
        # 実際の実装ではtorchaudio.functionalのresampleを使用することを推奨
        input_length = x.size(-1)
        target_length = math.ceil(input_length * self.resampling_ratio)
        return F.interpolate(x, size=target_length, mode="linear", align_corners=False)
