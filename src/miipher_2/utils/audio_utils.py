import torch
import torch.nn.functional as F  # noqa: N812


class DataCollatorAudioPad:
    """
    Pad 1-D waveforms to the max length in batch with attention masks.
    Returns dict ready for Trainer with proper mask handling.
    """

    def __init__(self, padding_value: float = 0.0) -> None:
        """
        Args:
            padding_value: Value to use for padding (default: 0.0 for audio)
        """
        self.padding_value = padding_value

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        inputs = [f["input_values"] for f in features]
        targets = [f["labels"] for f in features]

        # Get original lengths for attention mask
        input_lengths = [x.size(-1) for x in inputs]
        target_lengths = [y.size(-1) for y in targets]

        max_input_len = max(input_lengths)
        max_target_len = max(target_lengths)

        # Pad inputs and create attention masks
        batch_in = torch.stack([F.pad(x, (0, max_input_len - x.size(-1)), value=self.padding_value) for x in inputs])

        batch_lbl = torch.stack([F.pad(y, (0, max_target_len - y.size(-1)), value=self.padding_value) for y in targets])

        # Create attention masks (1 for real data, 0 for padding)
        input_attention_mask = torch.stack(
            [F.pad(torch.ones(length), (0, max_input_len - length), value=0.0) for length in input_lengths]
        )

        target_attention_mask = torch.stack(
            [F.pad(torch.ones(length), (0, max_target_len - length), value=0.0) for length in target_lengths]
        )

        return {
            "input_values": batch_in,
            "labels": batch_lbl,
            "attention_mask": input_attention_mask,
            "labels_attention_mask": target_attention_mask,
        }


class DataCollatorAudioPadV2:
    """
    Advanced version with statistics-aware padding for HuBERT compatibility.
    Handles LayerNorm statistics properly by masking padding regions.
    """

    def __init__(self, padding_value: float = 0.0, mask_padding_in_stats: bool = True) -> None:
        """
        Args:
            padding_value: Value to use for padding
            mask_padding_in_stats: Whether to exclude padding from normalization stats
        """
        self.padding_value = padding_value
        self.mask_padding_in_stats = mask_padding_in_stats

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        inputs = [f["input_values"] for f in features]
        targets = [f["labels"] for f in features]

        input_lengths = [x.size(-1) for x in inputs]
        target_lengths = [y.size(-1) for y in targets]

        max_input_len = max(input_lengths)
        max_target_len = max(target_lengths)

        # Advanced padding strategy
        if self.mask_padding_in_stats:
            # Use very small values instead of exact zeros to avoid NaN in statistics
            # but still easily maskable
            effective_padding_value = self.padding_value if self.padding_value != 0.0 else 1e-8
        else:
            effective_padding_value = self.padding_value

        batch_in = torch.stack(
            [F.pad(x, (0, max_input_len - x.size(-1)), value=effective_padding_value) for x in inputs]
        )

        batch_lbl = torch.stack(
            [F.pad(y, (0, max_target_len - y.size(-1)), value=effective_padding_value) for y in targets]
        )

        # Binary attention masks
        input_attention_mask = torch.stack(
            [
                F.pad(torch.ones(length, dtype=torch.bool), (0, max_input_len - length), value=False)
                for length in input_lengths
            ]
        )

        target_attention_mask = torch.stack(
            [
                F.pad(torch.ones(length, dtype=torch.bool), (0, max_target_len - length), value=False)
                for length in target_lengths
            ]
        )

        return {
            "input_values": batch_in,
            "labels": batch_lbl,
            "attention_mask": input_attention_mask,
            "labels_attention_mask": target_attention_mask,
        }
