import hydra
import torch
from omegaconf import DictConfig
from torchinfo import summary

from miipher_2.model.feature_cleaner import FeatureCleaner


@hydra.main(version_base=None, config_path="../configs", config_name="adapter_layer_6_mhubert_147")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Creating FeatureCleaner model...")
    model = FeatureCleaner(cfg.model)
    model.to(device)
    model.eval()

    print("=" * 40, "Model architecture", "=" * 40)
    print(model)
    print("=" * 80)

    # Sample input (16kHz audio, 3 seconds)
    batch_size = 2
    sequence_length = 48000  # 3 seconds at 16kHz
    input_shape = (batch_size, sequence_length)

    print("=" * 40, f"\nModel architecture summary (input shape: {input_shape})", "=" * 40)

    summary(
        model,
        input_size=input_shape,
        device=device,
        dtypes=[torch.float32],
        depth=3,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names"],
    )

    print("\n" + "=" * 80)
    print("Parameter counts:")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable ratio: {trainable_params / total_params:.2%}")

    # Show adapter-specific parameters
    adapter_params = 0
    for name, param in model.named_parameters():
        if "adapter" in name and param.requires_grad:
            adapter_params += param.numel()

    print(f"Adapter parameters: {adapter_params:,}")

    print("\n" + "=" * 80)
    print("Trainable modules:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape} ({param.numel():,} params)")


if __name__ == "__main__":
    main()
