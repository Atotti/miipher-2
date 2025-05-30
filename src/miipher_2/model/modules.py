import math

import torch
import torch.nn.functional as F
from torch import nn as nn


class FiLMLayer(nn.Module):
    def __init__(self, input_channels, intermediate_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, intermediate_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(intermediate_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        batch_size, K, D = a.size()
        Q = b.size(1)
        a = a.transpose(1, 2)
        output = self.conv2((self.leaky_relu(self.conv1(a)).transpose(1, 2) + b).transpose(1, 2))
        output = output.permute(0, 2, 1)
        assert output.size() == (batch_size, K, D)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        return self.pe[x]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Postnet(nn.Module):
    """Postnet
    - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(
        self,
        n_mel_channels,
        postnet_embedding_dim,
        postnet_kernel_size,
        postnet_n_convolutions,
    ):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        for i in range(len(self.convolutions) - 1):
            x = torch.nn.functional.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = torch.nn.functional.dropout(self.convolutions[-1](x), 0.5, self.training)
        x = x.transpose(1, 2)
        return x


class ParallelAdapter(nn.Module):
    """
    Parallel Adapter for parameter-efficient fine-tuning of USM

    This adapter is applied in parallel to each USM layer to predict
    clean features from noisy inputs with minimal additional parameters.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim)
        )

        # Initialize weights to small values for stable training
        self._init_weights()

    def _init_weights(self):
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, T, D)
        Returns:
            Adapter output to be added to input (B, T, D)
        """
        return self.ffn(x)


class MemoryEfficientWaveFit(nn.Module):
    """
    Memory-efficient WaveFit vocoder for Miipher-2

    Key improvements:
    1. Feature pre-upsampler to replace transposed convolutions
    2. Simplified FiLM layer and UBlock structure
    3. Reduced memory footprint for large-scale processing
    """

    def __init__(
        self,
        input_dim: int = 1532,
        sample_rate: int = 24000,
        hop_length: int = 300,  # 24000 / 80 = 300 for 80 Hz frame rate
    ):
        super().__init__()

        self.input_dim = input_dim
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        # Feature pre-upsampler to replace transposed convolutions
        self.feature_upsampler = FeatureUpsampler(input_dim, hop_length)

        # U-Net for waveform synthesis
        self.unet = MemoryEfficientUNet()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Synthesize waveform from USM features

        Args:
            features: USM features (B, T, D)
        Returns:
            waveform: Synthesized waveform (B, T * hop_length)
        """
        # Upsample features to match target waveform length
        upsampled_features = self.feature_upsampler(features)

        # Generate waveform using U-Net
        waveform = self.unet(upsampled_features)

        return waveform


class FeatureUpsampler(nn.Module):
    """
    Feature upsampler that repeats features to match waveform frame rate
    Replaces memory-intensive transposed convolutions
    """

    def __init__(self, input_dim: int, hop_length: int):
        super().__init__()
        self.hop_length = hop_length

        # Simple linear projection for feature dimension adjustment
        self.feature_proj = nn.Linear(input_dim, 128)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Input features (B, T, D)
        Returns:
            upsampled_features: Upsampled features (B, T * hop_length, 128)
        """
        B, T, D = features.shape

        # Project features
        features = self.feature_proj(features)  # (B, T, 128)

        # Repeat features along time dimension
        features = features.unsqueeze(2)  # (B, T, 1, 128)
        features = features.repeat(1, 1, self.hop_length, 1)  # (B, T, hop_length, 128)
        features = features.reshape(B, T * self.hop_length, 128)  # (B, T * hop_length, 128)

        return features


class MemoryEfficientUNet(nn.Module):
    """
    Memory-efficient U-Net for WaveFit with simplified FiLM layers
    """

    def __init__(self):
        super().__init__()

        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList(
            [
                DownBlock(128, 128, downsample_factor=2),
                DownBlock(128, 128, downsample_factor=2),
                DownBlock(128, 256, downsample_factor=3),
                DownBlock(256, 512, downsample_factor=4),
            ]
        )

        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(512, 512, upsample_factor=5),
                UpBlock(1024, 256, upsample_factor=4),  # 512 + 512 = 1024 from skip connection
                UpBlock(512, 128, upsample_factor=3),  # 256 + 256 = 512 from skip connection
                UpBlock(256, 128, upsample_factor=2),  # 128 + 128 = 256 from skip connection
                UpBlock(256, 128, upsample_factor=2),  # 128 + 128 = 256 from skip connection
            ]
        )

        # Final output layer
        self.output_layer = nn.Conv1d(128, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, T, 128)
        Returns:
            waveform: Output waveform (B, T)
        """
        # Convert to channel-first format for convolutions
        x = x.transpose(1, 2)  # (B, 128, T)

        # Encoder path with skip connections
        skip_connections = []
        for down_block in self.down_blocks:
            x = down_block(x)
            skip_connections.append(x)

        # Decoder path with skip connections
        for i, up_block in enumerate(self.up_blocks):
            if i > 0:  # Skip first up block which doesn't need skip connection
                skip = skip_connections[-(i)]
                x = torch.cat([x, skip], dim=1)
            x = up_block(x)

        # Output layer
        waveform = self.output_layer(x)  # (B, 1, T)
        waveform = waveform.squeeze(1)  # (B, T)

        return waveform


class DownBlock(nn.Module):
    """Downsampling block for U-Net encoder"""

    def __init__(self, in_channels: int, out_channels: int, downsample_factor: int):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool1d(kernel_size=downsample_factor, stride=downsample_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block for U-Net decoder"""

    def __init__(self, in_channels: int, out_channels: int, upsample_factor: int):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=upsample_factor, mode="nearest")
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
