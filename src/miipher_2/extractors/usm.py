import torch
from transformers import Gemma3nAudioEncoder, Gemma3nAudioFeatureExtractor


class USMExtractor(torch.nn.Module):
    def __init__(self, model_name: str, layer: int) -> None:
        super().__init__()
        self.fe = Gemma3nAudioFeatureExtractor.from_pretrained(model_name)
        self.model = Gemma3nAudioEncoder.from_pretrained("google/gemma-3n-e2b-it", output_hidden_states=True)
        self.layer = layer

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: (B, T) float32, 16 kHz, -1 ~ 1
        Returns:
            feat: (B, C, T/320) ? Hz
        """
        hs: list[torch.Tensor] = self.model(wav, output_hidden_states=True, return_dict=True).hidden_states
        return hs[self.layer + 1].transpose(1, 2).contiguous()
