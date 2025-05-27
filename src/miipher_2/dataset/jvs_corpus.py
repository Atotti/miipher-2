from pathlib import Path
from typing import Literal

import torchaudio
from torch.utils.data import Dataset


class JVSCorpus(Dataset):
    def __init__(self, root: str, exclude_speakers: tuple=()) -> None:
        super().__init__()
        self.root = Path(root)
        self.speakers = [f.stem for f in self.root.glob("jvs*") if f.is_dir() and f.stem not in exclude_speakers]
        self.clean_texts = {}
        self.wav_files = []
        for speaker in self.speakers:
            transcript_files = (self.root / speaker).glob("**/transcripts_utf8.txt")
            for transcript_file in transcript_files:
                subset = transcript_file.parent.name
                with transcript_file.open() as f:
                    lines = f.readlines()
                for line in lines:
                    wav_name, text = line.strip().split(":")
                    self.clean_texts[f"{speaker}/{subset}/{wav_name}"] = text
                    wav_path = self.root / Path(f"{speaker}/{subset}/wav24kHz16bit/{wav_name}.wav")
                    if wav_path.exists():
                        self.wav_files.append(wav_path)

    def __getitem__(self, index: int) -> dict[str, str]:
        wav_path = self.wav_files[index]
        wav_tensor, sr = torchaudio.load(wav_path)
        wav_path = wav_path.resolve()
        speaker = wav_path.parent.parent.parent.stem
        subset = wav_path.parent.parent.stem
        wav_name = wav_path.stem

        clean_text = self.clean_texts[f"{speaker}/{subset}/{wav_name}"]

        basename = f"{subset}_{speaker}_{wav_name}"
        return {
            "wav_path": str(wav_path),
            "speaker": speaker,
            "clean_text": clean_text,
            "basename": basename,
            "lang_code": "jpn",
        }

    def __len__(self) -> int:
        return len(self.wav_files)

    @property
    def speaker_dict(self) -> dict[str, int]:
        speakers = set()
        for wav_path in self.wav_files:
            speakers.add(wav_path.parent.parent.parent.stem)
        return {x: idx for idx, x in enumerate(speakers)}

    @property
    def lang_code(self) -> Literal["jpn"]:
        return "jpn"
