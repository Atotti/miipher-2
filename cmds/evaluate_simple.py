#!/usr/bin/env python3
"""
evaluate_simple.py
==================
ECAPAの話者性とDNSMOSの2指標でのみ評価
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition
from tqdm.auto import tqdm

# DNSMOS
from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore


def get_logger(name: str = "eval", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.setLevel(level)
    return logger


log = get_logger()


def load_wav(path: Path, sr: int):
    wav, s = torchaudio.load(path)
    if s != sr:
        wav = torchaudio.functional.resample(wav, s, sr)
    return wav


def speaker_cos_ecapa(ref: torch.Tensor, syn: torch.Tensor, ecapa_model) -> float:
    """ECAPAモデルを使用して話者類似度を計算"""
    if ref.dim() == 1:
        ref = ref.unsqueeze(0)
    if syn.dim() == 1:
        syn = syn.unsqueeze(0)

    device = next(ecapa_model.parameters()).device
    ref, syn = ref.to(device), syn.to(device)

    with torch.no_grad():
        emb_ref = ecapa_model.encode_batch(ref)
        emb_syn = ecapa_model.encode_batch(syn)

    # 次元調整
    while emb_ref.dim() > 1:
        emb_ref = emb_ref.mean(dim=0)
    while emb_syn.dim() > 1:
        emb_syn = emb_syn.mean(dim=0)

    sim = torch.nn.functional.cosine_similarity(emb_ref, emb_syn, dim=0, eps=1e-8)
    return float(sim)


def compute_dnsmos(wav: torch.Tensor, model) -> dict:
    """DNSMOSスコアを計算"""
    # DNSMOSは音声テンソルを受け取る（1次元で入力）
    if wav.dim() > 1:
        wav = wav.squeeze()
    
    with torch.no_grad():
        scores = model(wav)
    
    # DNSMOSは4つのスコアを返す: [p808_mos, mos_sig, mos_bak, mos_ovr]
    return {
        'p808_mos': float(scores[0]),
        'mos_sig': float(scores[1]),
        'mos_bak': float(scores[2]),
        'mos_ovr': float(scores[3])
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True, type=Path)
    ap.add_argument("--restored_dir", required=True, type=Path)
    ap.add_argument("--outfile", default=Path("eval_results_simple.csv"), type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    t0 = time.time()
    log.info("Loading models...")
    
    # ECAPAモデル
    ecapa = SpeakerRecognition.from_hparams(
        "speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": args.device},
        savedir=str(Path.home() / ".cache/speechbrain/ecapa"),
    )
    
    # DNSMOSモデル
    dnsmos_model = DeepNoiseSuppressionMeanOpinionScore(fs=args.sr, personalized=False)
    dnsmos_model = dnsmos_model.to(args.device)
    
    log.info(f"Models loaded in {time.time() - t0:.1f} s")

    rows = []
    clean_files = sorted(args.clean_dir.glob("**/*.wav"))
    tot = len(clean_files)
    log.info(f"Start evaluation on {tot} files")

    ecapa_scores = []
    dnsmos_scores = []

    for cl_path in tqdm(clean_files, desc="Evaluating"):
        rel = cl_path.relative_to(args.clean_dir)
        res_path = args.restored_dir / rel
        
        if not res_path.exists():
            warnings.warn(f"skip {rel} (missing restored file)", stacklevel=2)
            continue

        cl = load_wav(cl_path, args.sr)
        res = load_wav(res_path, args.sr)

        # ECAPA話者類似度
        ecapa_score = speaker_cos_ecapa(cl, res, ecapa)
        ecapa_scores.append(ecapa_score)

        # DNSMOSスコア
        res_device = res.to(args.device)
        dnsmos_score = compute_dnsmos(res_device, dnsmos_model)
        dnsmos_scores.append(dnsmos_score)

        rows.append({
            "file": str(rel),
            "ECAPA_cos": ecapa_score,
            "DNSMOS_p808": dnsmos_score['p808_mos'],
            "DNSMOS_sig": dnsmos_score['mos_sig'],
            "DNSMOS_bak": dnsmos_score['mos_bak'],
            "DNSMOS_ovr": dnsmos_score['mos_ovr'],
        })

    # 結果をCSVに保存
    df = pd.DataFrame(rows)
    df.to_csv(args.outfile, index=False)
    log.info(f"CSV saved to {args.outfile}")

    # 平均値の表示
    log.info("\n=== Evaluation Results (Mean) ===")
    log.info(f"ECAPA Speaker Similarity: {sum(ecapa_scores) / len(ecapa_scores):.4f}")
    
    # DNSMOS各スコアの平均値を計算
    avg_p808 = sum(s['p808_mos'] for s in dnsmos_scores) / len(dnsmos_scores)
    avg_sig = sum(s['mos_sig'] for s in dnsmos_scores) / len(dnsmos_scores)
    avg_bak = sum(s['mos_bak'] for s in dnsmos_scores) / len(dnsmos_scores)
    avg_ovr = sum(s['mos_ovr'] for s in dnsmos_scores) / len(dnsmos_scores)
    
    log.info(f"DNSMOS P808: {avg_p808:.4f}")
    log.info(f"DNSMOS Signal: {avg_sig:.4f}")
    log.info(f"DNSMOS Background: {avg_bak:.4f}")
    log.info(f"DNSMOS Overall: {avg_ovr:.4f}")
    log.info("=================================")


if __name__ == "__main__":
    main()