import io
import os
import pathlib

import hydra
import torch
import torchaudio
import tqdm
import webdataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from miipher_2.preprocess import DegradationApplier


class Preprocessor:
    """
    Preprocess dataset
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Args:
            cfg: hydra config
        """
        self.cfg = cfg
        self.dataset = hydra.utils.instantiate(cfg.preprocess.preprocess_dataset)
        self.sampling_rate = self.cfg.sample_rate
        self.degradation_model = DegradationApplier(cfg.preprocess.degradation)
        self.text2phone_dict = {}
        self.n_repeats = cfg.preprocess.n_repeats

    @torch.inference_mode()  # type: ignore
    def process_utterance(
        self,
        basename: str,
        audio_file_path: pathlib.Path,
        lang_code: str,
    ) -> list[dict[str, bytes | str]]:
        orig_waveform, sample_rate = torchaudio.load(audio_file_path)

        waveform: torch.Tensor = torchaudio.functional.resample(
            orig_waveform, sample_rate, new_freq=self.sampling_rate
        )[0]  # remove channel dimension only support mono

        with audio_file_path.open(mode="rb") as f:
            wav_bytes = f.read()
        samples: list[dict[str, bytes | str]] = []
        for i in range(self.n_repeats):
            degraded_speech = self.apply_noise(waveform)
            buff = io.BytesIO()
            torchaudio.save(
                buff,
                src=degraded_speech.unsqueeze(0),
                sample_rate=self.sampling_rate,
                format="wav",
            )
            buff.seek(0)

            sample = {
                "__key__": basename + f"_{i}",
                "speech.wav": wav_bytes,
                "degraded_speech.wav": buff.read(),
                "resampled_speech.pth": webdataset.torch_dumps(waveform),
            }
            samples.append(sample)
        return samples

    def apply_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.degradation_model.process(waveform, self.sampling_rate)

    def build_from_path(self) -> None:
        pathlib.Path("/".join(self.cfg.preprocess.train_tar_sink.pattern.split("/")[:-1])).mkdir(exist_ok=True)
        train_sink = hydra.utils.instantiate(self.cfg.preprocess.train_tar_sink)
        val_sink = hydra.utils.instantiate(self.cfg.preprocess.val_tar_sink)
        num_workers: int = os.cpu_count() if os.cpu_count() is not None else 8
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=num_workers)
        for idx, data in enumerate(tqdm.tqdm(dataloader)):
            basename = data["basename"][0]
            wav_path = data["wav_path"][0]
            lang_code = data["lang_code"][0]
            result = self.process_utterance(basename, pathlib.Path(wav_path), lang_code)
            sink = train_sink if idx >= self.cfg.preprocess.val_size else val_sink
            for sample in result:
                sink.write(sample)
        train_sink.close()
        val_sink.close()
