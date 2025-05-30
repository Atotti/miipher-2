"""
Neural network modules for the Miipher-2 model.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class FiLMLayer(nn.Module):
    def __init__(self, input_channels: int, intermediate_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, intermediate_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(intermediate_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        batch_size, K, D = a.size()
        Q = b.size(1)
        a = a.transpose(1, 2)
        output = self.conv2((self.leaky_relu(self.conv1(a)).transpose(1, 2) + b).transpose(1, 2))
        output = output.permute(0, 2, 1)
        assert output.size() == (batch_size, K, D)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
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
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear_2(x)


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        bias: bool = True,
        w_init_gain: str = "linear",
    ) -> None:
        super().__init__()
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

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.conv(signal)


class Postnet(nn.Module):
    """
    Postnet
    - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(
        self,
        n_mel_channels: int,
        postnet_embedding_dim: int,
        postnet_kernel_size: int,
        postnet_n_convolutions: int,
    ) -> None:
        super().__init__()
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

        for _ in range(1, postnet_n_convolutions - 1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))
            x = torch.nn.functional.dropout(x, 0.5, self.training)
        x = torch.nn.functional.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x.transpose(1, 2)


class ParallelAdapter(nn.Module):
    """
    Parallel Adapter for parameter-efficient fine-tuning of USM

    This adapter is applied in parallel to each USM layer to predict
    clean features from noisy inputs with minimal additional parameters.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim)
        )

        # Initialize weights to small values for stable training
        self._init_weights()

    def _init_weights(self) -> None:
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
    ) -> None:
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
        return self.unet(upsampled_features)


class FeatureUpsampler(nn.Module):
    """
    Feature upsampler that repeats features to match waveform frame rate
    Replaces memory-intensive transposed convolutions
    """

    def __init__(self, input_dim: int, hop_length: int) -> None:
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
        return features.reshape(B, T * self.hop_length, 128)  # (B, T * hop_length, 128)


class MemoryEfficientUNet(nn.Module):
    """
    Memory-efficient U-Net for WaveFit with simplified FiLM layers
    """

    def __init__(self) -> None:
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
        return waveform.squeeze(1)  # (B, T)


class DownBlock(nn.Module):
    """Downsampling block for U-Net encoder"""

    def __init__(self, in_channels: int, out_channels: int, downsample_factor: int) -> None:
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool1d(kernel_size=downsample_factor, stride=downsample_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return self.downsample(x)


class UpBlock(nn.Module):
    """Upsampling block for U-Net decoder"""

    def __init__(self, in_channels: int, out_channels: int, upsample_factor: int) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=upsample_factor, mode="nearest")
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


class LocationLayer(nn.Module):
    """Location layer for attention mechanism."""

    def __init__(self, attention_n_filters: int, attention_kernel_size: int, attention_dim: int) -> None:
        super().__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            2, attention_n_filters, kernel_size=attention_kernel_size, padding=padding, bias=False, stride=1, dilation=1
        )
        self.location_dense = nn.Linear(attention_n_filters, attention_dim)

    def forward(self, attention_weights_cat: torch.Tensor) -> torch.Tensor:
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        return self.location_dense(processed_attention)


class Attention(nn.Module):
    """Attention module for sequence-to-sequence models."""

    def __init__(
        self,
        attention_rnn_dim: int,
        embedding_dim: int,
        attention_dim: int,
        attention_location_n_filters: int,
        attention_location_kernel_size: int,
    ) -> None:
        super().__init__()
        self.query_layer = nn.Linear(attention_rnn_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters, attention_location_kernel_size, attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(
        self, query: torch.Tensor, processed_memory: torch.Tensor, attention_weights_cat: torch.Tensor
    ) -> torch.Tensor:
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_memory))

        return energies.squeeze(-1)

    def forward(
        self,
        attention_hidden_state: torch.Tensor,
        memory: torch.Tensor,
        processed_memory: torch.Tensor,
        attention_weights_cat: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = torch.nn.functional.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)

        return attention_context.squeeze(1), attention_weights


class PhoneFeatureProjection(nn.Module):
    """
    Projects phone features to a different dimensionality and optionally adds noise.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, phone_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phone_features: Input phone features (B, T, input_dim)

        Returns:
            Projected phone features (B, T, output_dim)
        """
        return self.ffn(phone_features)


class Miipher2Vocoder(nn.Module):
    """
    Miipher-2 vocoder that converts semantic features to waveform.

    This vocoder consists of:
    1. Feature upsampling
    2. U-Net based waveform generation
    """

    def __init__(
        self,
        input_dim: int = 1532,
        hop_length: int = 256,
    ) -> None:
        super().__init__()
        self.feature_upsampler = FeatureUpsampler(input_dim, hop_length)
        self.unet = UNetWaveformGenerator()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Input features (B, T, input_dim)

        Returns:
            waveform: Generated waveform (B, T * hop_length)
        """
        # Upsample features to match waveform resolution
        upsampled_features = self.feature_upsampler(features)

        # Generate waveform using U-Net
        return self.unet(upsampled_features)


class UNetWaveformGenerator(nn.Module):
    """
    U-Net architecture for waveform generation.
    """

    def __init__(self) -> None:
        super().__init__()

        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList(
            [
                DownBlock(128, 256, 4),  # (B, T//4, 256)
                DownBlock(256, 512, 4),  # (B, T//16, 512)
                DownBlock(512, 1024, 4),  # (B, T//64, 1024)
            ]
        )

        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(1024, 512, 4),  # (B, T//16, 512)
                UpBlock(512, 256, 4),  # (B, T//4, 256)
                UpBlock(256, 128, 4),  # (B, T, 128)
            ]
        )

        # Final convolution to produce waveform
        self.final_conv = nn.Conv1d(128, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Upsampled features (B, T, 128)

        Returns:
            waveform: Generated waveform (B, T)
        """
        # Transpose for conv1d: (B, T, 128) -> (B, 128, T)
        x = features.transpose(1, 2)

        # Encoder path with skip connections
        skip_connections = []
        for down_block in self.down_blocks:
            skip_connections.append(x)
            x = down_block(x)

        # Decoder path with skip connections
        for i, up_block in enumerate(self.up_blocks):
            if i > 0:  # Skip connection for all except first decoder layer
                skip = skip_connections[-(i + 1)]
                # Ensure matching dimensions for concatenation
                if x.size(2) != skip.size(2):
                    skip = torch.nn.functional.interpolate(skip, size=x.size(2), mode="linear", align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = up_block(x)

        # Final convolution
        waveform = self.final_conv(x)
        return waveform.squeeze(1)  # (B, T)


class DownBlock(nn.Module):
    """Downsampling block for U-Net encoder"""

    def __init__(self, in_channels: int, out_channels: int, downsample_factor: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.downsample = nn.Conv1d(
            out_channels, out_channels, kernel_size=downsample_factor, stride=downsample_factor, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return self.downsample(x)


class UpBlock(nn.Module):
    """Upsampling block for U-Net decoder"""

    def __init__(self, in_channels: int, out_channels: int, upsample_factor: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=upsample_factor, stride=upsample_factor, padding=0
        )
        self.conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)
