import torch
from transformers import HubertModel, Wav2Vec2Model


class SSLExtractor(torch.nn.Module):
    def __init__(self, model_name: str, layer: int, model_type: str = "auto") -> None:
        super().__init__()
        self.layer = layer
        self.model_type = model_type

        if model_type == "auto":
            if "hubert" in model_name.lower():
                model_type = "hubert"
            elif "wav2vec2" in model_name.lower():
                model_type = "wav2vec2"
            elif "wavlm" in model_name.lower():
                model_type = "hubert"  # WavLM is based on HuBERT
            else:
                msg = f"Cannot auto-detect model type for {model_name}"
                raise ValueError(msg)

        if model_type == "hubert":
            self.model = HubertModel.from_pretrained(model_name, output_hidden_states=True)
        elif model_type == "wav2vec2":
            self.model = Wav2Vec2Model.from_pretrained(model_name, output_hidden_states=True)
        else:
            msg = f"Unsupported model type: {model_type}"
            raise ValueError(msg)

        self.model_type = model_type

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: (B, T) float32, 16 kHz, -1 ~ 1
        Returns:
            feat: (B, C, T/320) 50 Hz
        """
        outputs = self.model(wav, output_hidden_states=True, return_dict=True)
        hs: list[torch.Tensor] = outputs.hidden_states
        # 指定された層を転置して返す
        return hs[self.layer + 1].transpose(1, 2).contiguous()


# 後方互換性のためのエイリアス
class HubertExtractor(SSLExtractor):
    def __init__(self, model_name: str, layer: int) -> None:
        super().__init__(model_name, layer, model_type="hubert")
