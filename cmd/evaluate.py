#!/usr/bin/env python3
"""
evaluate_speech_restoration.py
==============================
Clean / Degraded / Restored から指標を算出し CSV 出力
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from tqdm.auto import tqdm

import miipher_2.utils.eval_utils as U

log = U.get_logger("eval")


def load_wav(path: Path, sr: int):
    wav, s = torchaudio.load(path)
    if s != sr:
        wav = torchaudio.functional.resample(wav, s, sr)
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True, type=Path)
    ap.add_argument("--degraded_dir", required=True, type=Path)
    ap.add_argument("--restored_dir", required=True, type=Path)
    ap.add_argument("--outfile", default=Path("eval_results.csv"), type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--log_every", type=int, default=10, help="log every N files")
    args = ap.parse_args()

    t0 = time.time()
    log.info("Loading ASR & speaker models (this may take a while)…")
    asr_model, asr_proc = U.load_asr(args.device)
    xvec, ecapa = U.load_spk_models(args.device)
    log.info(f"Models loaded in {time.time() - t0:.1f} s")

    rows, start = [], time.time()
    clean_files = sorted(args.clean_dir.glob("**/*.wav"))
    tot = len(clean_files)
    log.info(f"Start evaluation on {tot} files")

    for i, cl_path in enumerate(clean_files, 1):
        rel = cl_path.relative_to(args.clean_dir)
        deg_path = args.degraded_dir / rel
        res_path = args.restored_dir / rel
        if not deg_path.exists() or not res_path.exists():
            warnings.warn(f"skip {rel} (missing degraded/restored)")
            continue

        cl, deg, res = (load_wav(p, args.sr) for p in (cl_path, deg_path, res_path))

        rows.append(
            dict(
                file=str(rel),
                MCD=U.mcd(cl, res, args.sr),
                XvecCos=U.speaker_cos(cl.to(args.device), res.to(args.device), args.sr, xvec),
                ECAPACos=U.speaker_cos(cl.to(args.device), res.to(args.device), args.sr, ecapa),
                WER=U.asr_wer(cl.to(args.device), res.to(args.device), args.sr, asr_model, asr_proc, args.device),
                logF0_RMSE=U.logf0_rmse(cl, res, args.sr),
                # 劣化比較
                Deg_MCD=U.mcd(cl, deg, args.sr),
                Deg_XvecCos=U.speaker_cos(cl.to(args.device), deg.to(args.device), args.sr, xvec),
                Deg_ECAPACos=U.speaker_cos(cl.to(args.device), deg.to(args.device), args.sr, ecapa),
                Deg_WER=U.asr_wer(cl.to(args.device), deg.to(args.device), args.sr, asr_model, asr_proc, args.device),
                Deg_logF0_RMSE=U.logf0_rmse(cl, deg, args.sr),
            )
        )

        if i % args.log_every == 0 or i == tot:
            elapsed = time.time() - start
            log.info(f"{i}/{tot} files evaluated  |  elapsed {elapsed / 60:.1f} min")

    df = pd.DataFrame(rows)
    df.to_csv(args.outfile, index=False)
    log.info(f"CSV saved to {args.outfile}")
    log.info(df.describe().loc[["mean", "std", "min", "max"]].to_string())
    log.info("Done")


if __name__ == "__main__":
    main()
