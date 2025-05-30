from collections.abc import Iterable
from typing import Callable

import torch
from peft import inject_adapter_in_model
from torch import nn
from transformers import HubertModel

from miipher_2.model import ParallelAdapter, ParallelAdapterConfig, SpeechBrainHiFiGAN


# ---------------------------------------------------------------------------
# Miipher-2 main model (frozen HuBERT + Parallel Adapters + HiFi-GAN)
# ---------------------------------------------------------------------------
class Miipher2(nn.Module):
    """Miipher-2 - Universal Speech Restoration model.

    Parameters
    ----------
    usm_model_name : str
        HF hub ID of the frozen feature extractor (HuBERT large by default).
    target_layer : int
        0-indexed encoder layer whose hidden state will be cleaned / projected.
    reduction_factor : int
        Bottleneck ratio r (hidden // r).
    use_vocoder : bool
        Whether to include the SpeechBrain HiFi-GAN into the graph.
    mel_dim : int
        Dimensionality of the (log-)Mel projection given to HiFi-GAN.
    device : str | torch.device
        Device to place models / adapters.
    """

    def __init__(
        self,
        usm_model_name: str = "rinna/japanese-hubert-large",
        target_layer: int = 12,  # → layer-13 in paper (0-indexed)
        reduction_factor: int = 8,
        use_vocoder: bool = False,  # noqa: FBT001, FBT002
        mel_dim: int = 80,
    ) -> None:
        super().__init__()

        # 1  Load & freeze HuBERT
        backbone: HubertModel = HubertModel.from_pretrained(usm_model_name).to(self.device)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)  # noqa: FBT003
        self.backbone = backbone
        self.hidden_size: int = backbone.config.hidden_size  # 1024
        self.target_layer: int = target_layer

        # 2  Build Parallel Adapter config
        pa_cfg = ParallelAdapterConfig(reduction_factor=reduction_factor)

        # 3  Inject adapters into every FFN
        self._adapter_params: list[nn.Parameter] = []
        for _idx, layer in enumerate(backbone.encoder.layers):
            pa = self._attach_parallel_adapter(layer, pa_cfg)
            self._adapter_params.extend(pa.parameters())
        # register with PEFT (enables save_pretrained / from_pretrained)
        inject_adapter_in_model(pa_cfg, backbone, adapter_name="miipher2_pa")

        # 4  Projection to Mel space (1*1 conv == Linear)
        self.proj = nn.Linear(self.hidden_size, mel_dim, bias=False)

        # 5  HiFi-GAN vocoder
        self.use_vocoder = use_vocoder
        self.hifigan = SpeechBrainHiFiGAN(model_id="speechbrain/hifigan-hubert-k1000-LibriTTS", device=self.device)


    # ---------------------------------------------------------------------
    #  Helper - attach PA to one Transformer layer
    def _attach_parallel_adapter(self, layer: nn.Module, pa_cfg: ParallelAdapterConfig) -> ParallelAdapter:
        pa = ParallelAdapter(self.hidden_size, pa_cfg).to(self.device)

        original_ffn = layer.feed_forward  # save reference

        def _ffn_plus_adapter(
            x: torch.Tensor,
            *,
            orig: Callable[[torch.Tensor], torch.Tensor] = original_ffn,
            pa_mod: ParallelAdapter = pa,
        ) -> torch.Tensor:
            return orig(x) + pa_mod(x)

        layer.feed_forward = _ffn_plus_adapter  # dynamic patch
        return pa

    # ------------------------------------------------------------------
    #  Public helpers - retrieve trainable params only, etc.
    # ------------------------------------------------------------------
    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """Return an iterator over PA (+ projection + vocoder) parameters."""
        yield from self._adapter_params
        yield from self.proj.parameters()
        if self.use_vocoder and self.hifigan is not None:
            yield from self.hifigan.parameters()

    # ------------------------------------------------------------------
    #  Forward / inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_features(self, wav: torch.Tensor) -> torch.Tensor:  # (B, T)
        """Return clean latent features at *target_layer* after PA correction."""
        return self.backbone(wav, output_hidden_states=True).hidden_states[self.target_layer]

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav -> (clean) wav or latent depending on `use_vocoder`."""
        wav = wav.to(self.device)
        feats = self.extract_features(wav)  # (B, T', D)

        if not self.use_vocoder:
            return feats  # type: ignore[return-value]

        mel = self.proj(feats)  # (B, T', mel_dim)
        return self.hifigan(mel)  # (B, T)

    # convenience property ------------------------------------------------
    @property
    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())


# ---------------------------------------------------------------------------
# 4.  Quick sanity check (run as `python miipher2.py`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = Miipher2(use_vocoder=False, device="cpu")
    print("Trainable parameters :", model.num_trainable / 1e6, "M")

    # dummy forward
    dummy_wav = torch.randn(1, 16000)  # 1-s @ 16kHz (HuBERT expects 16k)
    with torch.no_grad():
        out = model(dummy_wav)
    print("Output shape:", out.shape)  # (B, T', D)
