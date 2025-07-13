import argparse
import itertools
import random
import time
from pathlib import Path

import torch
import torchaudio
from tqdm.auto import tqdm

from miipher_2.utils.eval_utils import degrade_waveform, get_logger

log = get_logger("degrade")


def load_noises(noise_dir: Path, sr: int = 16000):
    pool = []
    for p in itertools.islice(noise_dir.glob("**/*.*"), 300):
        try:
            wav, s = torchaudio.load(p)
            if s != sr:
                wav = torchaudio.functional.resample(wav, s, sr)
            pool.append(wav)
        except Exception:
            continue
    if not pool:
        msg = "noise_dir から有効な wav が読み込めませんでした"
        raise RuntimeError(msg)
    log.info(f"Loaded {len(pool)} noise files")
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True, type=Path)
    ap.add_argument("--noise_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    noises = load_noises(args.noise_dir, args.sr)

    clean_files = sorted(args.clean_dir.glob("*.wav"))
    tot = len(clean_files)
    log.info(f"Start degrading {tot} wav files")

    t0 = time.time()
    for i, wav_path in enumerate(clean_files, 1):
        wav, sr_ = torchaudio.load(wav_path)
        if sr_ != args.sr:
            wav = torchaudio.functional.resample(wav, sr_, args.sr)

        degraded = degrade_waveform(wav, args.sr, noises)

        out_path = args.out_dir / wav_path.relative_to(args.clean_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(out_path, degraded, args.sr, encoding="PCM_S", bits_per_sample=16)

        if i % 10 == 0 or i == tot:
            elapsed = time.time() - t0
            log.info(f"{i}/{tot} files done  |  elapsed {elapsed / 60:.1f} min")

    log.info("All files degraded successfully")


if __name__ == "__main__":
    main()
