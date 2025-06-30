from pathlib import Path


class LibriTTSRCorpus:
    def __init__(self, root: str):
        self.root = Path(root)
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        splits = [
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
        ]

        for split in splits:
            split_path = self.root / split
            if not split_path.exists():
                continue

            for speaker_dir in split_path.glob("*"):
                if not speaker_dir.is_dir():
                    continue

                speaker_id = speaker_dir.name

                for chapter_dir in speaker_dir.glob("*"):
                    if not chapter_dir.is_dir():
                        continue

                    chapter_id = chapter_dir.name

                    # Load transcription file
                    trans_file = chapter_dir / f"{speaker_id}_{chapter_id}.trans.tsv"
                    if not trans_file.exists():
                        continue

                    # Parse transcription file
                    transcriptions = {}
                    with trans_file.open(encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split("\t")
                            if len(parts) >= 3:
                                utterance_id = parts[0]
                                normalized_text = parts[2]  # Use normalized text (3rd column)
                                transcriptions[utterance_id] = normalized_text

                    # Find corresponding wav files
                    for utterance_id, text in transcriptions.items():
                        wav_path = chapter_dir / f"{utterance_id}.wav"
                        if wav_path.exists():
                            self.samples.append(
                                {
                                    "wav_path": str(wav_path),
                                    "speaker": speaker_id,
                                    "clean_text": text,
                                    "basename": utterance_id,
                                    "lang_code": "eng",
                                }
                            )

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def speaker_dict(self) -> dict[str, int]:
        speakers = sorted(set(sample["speaker"] for sample in self.samples))
        return {speaker: idx for idx, speaker in enumerate(speakers)}
