import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor


class HubertExtractor(torch.nn.Module):
    def __init__(self, model_name: str, layer: int) -> None:
        super().__init__()
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.hubert = HubertModel.from_pretrained(model_name, output_hidden_states=True)
        self.layer = layer

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: (B, T) float32, 16 kHz, -1 ~ 1
        Returns:
            feat: (B, C, T/320) 50 Hz
        """
        hs: list[torch.Tensor] = self.hubert(wav, output_hidden_states=True, return_dict=True).hidden_states
        # 指定された層を転置して返す
        return hs[self.layer + 1].transpose(1, 2).contiguous()
