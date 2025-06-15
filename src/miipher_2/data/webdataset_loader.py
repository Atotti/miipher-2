import torchaudio
import webdataset as wds
from torch.utils.data import IterableDataset

SR16 = 16_000


class WavPairDataset(IterableDataset):
    """
    yield: noisy(1,T), clean(1,T) – both 16 kHz mono tensors
    """

    def __init__(self, pattern: str, shuffle: int = 1000) -> None:
        self.dataset = wds.WebDataset(pattern, resampled=True).shuffle(shuffle).decode(wds.torch_audio)

    def __iter__(self):
        for sample in self.dataset:
            clean, sr = sample["speech.wav"]
            noisy, _ = sample["degraded_speech.wav"]
            if sr != SR16:
                clean = torchaudio.functional.resample(clean, sr, SR16)
                noisy = torchaudio.functional.resample(noisy, sr, SR16)
            yield noisy.mean(0, keepdim=True), clean.mean(0, keepdim=True)
