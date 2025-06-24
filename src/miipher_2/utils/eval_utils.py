"""
audio_eval_utils.py
===================
* 劣化パイプライン
* 指標計算（MCD / X‑vector / ECAPA / WER / log‑F0‑RMSE）
  └ ASR・話者モデルは **必要時にのみ動的インポート** してロード時間を短縮
* 共通 Logger utility
"""

from __future__ import annotations

import io

# ───────────────────────── logging helper ──────────────────────────
import logging
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pysptk
import pyworld as pw
import soundfile as sf
import torch
import torchaudio


def get_logger(name: str = "audio_eval", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.setLevel(level)
    return logger


log = get_logger(__name__)

# ────────────────────── 劣化パイプライン ↓ ──────────────────────


def _encode_with_codec(wave: torch.Tensor, sr: int, fmt: str) -> torch.Tensor:
    assert fmt in ("mp3", "ogg")
    with tempfile.TemporaryDirectory() as d:
        raw = Path(d, "raw.wav")
        torchaudio.save(str(raw), wave, sr, encoding="PCM_S", bits_per_sample=16)
        coded = Path(d, f"coded.{fmt}")
        codec = "libmp3lame" if fmt == "mp3" else "libvorbis"
        subprocess.run(
            ["ffmpeg", "-loglevel", "quiet", "-y", "-i", raw, "-codec:a", codec, "-b:a", "64k", coded],
            check=True,
        )
        rec, sr2 = torchaudio.load(str(coded))
        assert sr2 == sr
        return rec


def _apply_reverb(wave: torch.Tensor, sr: int) -> torch.Tensor:
    wav, _ = torchaudio.sox_effects.apply_effects_tensor(wave, sr, [["reverb", "50", "50", "100"]])
    return wav


def _mix_noise(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    if noise.size(1) < clean.size(1):
        r = math.ceil(clean.size(1) / noise.size(1))
        noise = noise.repeat(1, r)[:, : clean.size(1)]
    else:
        s = random.randint(0, noise.size(1) - clean.size(1))
        noise = noise[:, s : s + clean.size(1)]

    pow_c = clean.pow(2).mean()
    pow_n = noise.pow(2).mean()
    scale = math.sqrt((pow_c / (10 ** (snr_db / 10))) / (pow_n + 1e-12))
    return clean + noise * scale


def degrade_waveform(clean: torch.Tensor, sr: int, noise_pool: list[torch.Tensor]) -> torch.Tensor:
    wav = clean.clone()
    wav = _encode_with_codec(wav, sr, random.choice(["mp3", "ogg"]))
    if random.random() < 0.5:
        wav = _apply_reverb(wav, sr)
    wav = _mix_noise(wav, random.choice(noise_pool), random.uniform(5, 30))
    peak = wav.abs().max()
    if peak > 0:
        wav *= 0.89 / peak
    return wav


def _next_pow2(x: int) -> int:
    """x 以上の最小 2 の冪."""
    return 1 << (x - 1).bit_length()


def _mcep_feat(
    x: np.ndarray,
    sr: int,
    order: int = 24,
    frame_shift_ms: float = 5.0,
    frame_len_ms: float = 25.0,
) -> np.ndarray:
    """
    WORLD 系 MCD 計算用メルケプストラム系列抽出.

    Parameters
    ----------
    x : np.ndarray
        1‑D waveform (float64 推奨, range ‑1…1)
    sr : int
        Sample rate (e.g. 16000)
    order : int, optional
        メルケプストラム次数 (default 24 → 25 次元; いわゆる MCD‑13 なら 12)
    frame_shift_ms, frame_len_ms : float
        フレーム長 / シフト (ms)

    Returns
    -------
    mc : ndarray, shape (num_frames, order+1)
    """

    # ---------- 定数 ----------
    alpha = 0.42 if sr == 16000 else 0.455  # 22.05 kHz → 0.455 等
    hop = int(frame_shift_ms * 0.001 * sr)
    win = int(frame_len_ms * 0.001 * sr)
    nfft = _next_pow2(win * 2)  # 2 の冪でないと SPTK が落ちる

    # ---------- 窓関数 ----------
    try:
        from pysptk.window import hamming  # ≥ 0.3

        w = hamming(win)
    except (ImportError, AttributeError):
        w = np.hamming(win)  # NumPy fallback

    # ---------- フレーム逐次処理 ----------
    feats: list[np.ndarray] = []
    err_frames = 0

    for i in range(0, len(x) - win, hop):
        frame = x[i : i + win] * w
        # 低振幅時に log(0)→ -inf を避けるため +1e‑12
        psd = np.abs(np.fft.rfft(frame, n=nfft)) ** 2 + 1e-12

        try:
            mc = pysptk.mcep(
                psd,
                order=order,
                alpha=alpha,
                maxiter=100,
                etype=1,  # log magnitude spectral distortion
            )
        except RuntimeError:
            # 無音 or 発散で mcep 失敗 → ゼロベクトルを採用
            err_frames += 1
            mc = np.zeros(order + 1, dtype=np.float64)

        feats.append(mc)

    if err_frames:
        log.debug("mcep: %d frames fell back to zeros (%.2f%%)", err_frames, 100 * err_frames / max(len(feats), 1))

    if not feats:
        # クリッピング等で 1 フレームも取れない場合の保険
        return np.zeros((1, order + 1), dtype=np.float64)

    return np.vstack(feats)


def _to_mono_1d_numpy(wave: torch.Tensor) -> np.ndarray:
    """PyTorch テンソルを、pyworld が要求するモノラル/1D/float64/C-contiguous な NumPy 配列に変換する"""
    # チャンネル次元があれば平均化してモノラルにする
    if wave.dim() > 1 and wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)

    # NumPy 配列に変換し、型とメモリ配置を整える
    np_wave = wave.squeeze().cpu().numpy().astype(np.float64)
    return np.ascontiguousarray(np_wave)


def mcd(ref: torch.Tensor, syn: torch.Tensor, sr: int) -> float:
    ref_np = _to_mono_1d_numpy(ref)
    syn_np = _to_mono_1d_numpy(syn)

    r, s = _mcep_feat(ref_np, sr), _mcep_feat(syn_np, sr)

    L = min(len(r), len(s))
    diff = (r[:L] - s[:L]) ** 2
    return (10 / np.log(10)) * np.sqrt(2 * diff.sum(axis=1)).mean()


def logf0_rmse(ref: torch.Tensor, syn: torch.Tensor, sr: int) -> float:
    ref_np = _to_mono_1d_numpy(ref)
    syn_np = _to_mono_1d_numpy(syn)

    fr, _ = pw.harvest(ref_np, sr)
    fs, _ = pw.harvest(syn_np, sr)

    L = min(len(fr), len(fs))
    mask = np.logical_and(fr[:L] > 0, fs[:L] > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.log(fr[:L][mask]) - np.log(fs[:L][mask])) ** 2)))


# ─── 以下は必要時ロード ──────────────────────────────────────────
def _lazy_import_speechbrain():
    global SpeakerRecognition
    from speechbrain.pretrained import SpeakerRecognition  # type: ignore

    return SpeakerRecognition


def _lazy_import_transformers():
    global AutoProcessor, AutoModelForSpeechSeq2Seq
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor  # type: ignore

    return AutoProcessor, AutoModelForSpeechSeq2Seq


def speaker_cos(
    ref: torch.Tensor,
    syn: torch.Tensor,
    sr: int,
    recognizer,
) -> float:
    """
    Robust cosine‑similarity (scalar) between two utterances.

    * 正常系: 返り値 ∈ [‑1, 1]
    * エンベディング形状が (B, T, D) / (B, D) / (D,) いずれでも動く
    """

    # -------- waveform shape → (1, T) --------
    if ref.dim() == 1:
        ref = ref.unsqueeze(0)
    if syn.dim() == 1:
        syn = syn.unsqueeze(0)

    device = next(recognizer.parameters()).device
    ref, syn = ref.to(device), syn.to(device)

    with torch.no_grad():
        emb_ref = recognizer.encode_batch(ref)  # shape: (⋯, D)
        emb_syn = recognizer.encode_batch(syn)

    # -------- すべての非最終次元を平均 --------
    #   (B, D) -> (D,) / (B, T, D) -> (D,) / (D,) -> (D,)
    while emb_ref.dim() > 1:
        emb_ref = emb_ref.mean(dim=0)
    while emb_syn.dim() > 1:
        emb_syn = emb_syn.mean(dim=0)

    # -------- cosine → scalar --------
    sim = torch.nn.functional.cosine_similarity(emb_ref, emb_syn, dim=0, eps=1e-8)  # ==> shape ()
    return float(sim)


def load_spk_models(device: str):
    SpeakerRecognition = _lazy_import_speechbrain()
    x = SpeakerRecognition.from_hparams(
        "speechbrain/spkrec-xvect-voxceleb",
        run_opts={"device": device},
        savedir=str(Path.home() / ".cache/speechbrain/xvect"),
    )
    e = SpeakerRecognition.from_hparams(
        "speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir=str(Path.home() / ".cache/speechbrain/ecapa"),
    )
    return x, e


def load_asr(device: str):
    AutoProcessor, AutoModelForSpeechSeq2Seq = _lazy_import_transformers()
    proc = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
    ).to(device)
    model.eval()
    return model, proc


def asr_wer(ref: torch.Tensor, syn: torch.Tensor, sr: int, model, proc, device: str) -> float:
    with torch.no_grad():
        ir = proc(ref.squeeze().cpu().numpy(), sampling_rate=sr, return_tensors="pt")
        isyn = proc(syn.squeeze().cpu().numpy(), sampling_rate=sr, return_tensors="pt")

        model_dtype = next(model.parameters()).dtype
        ir = ir.to(device)
        ir["input_features"] = ir["input_features"].to(model_dtype)
        isyn = isyn.to(device)
        isyn["input_features"] = isyn["input_features"].to(model_dtype)

        tr = proc.decode(model.generate(**ir)[0])
        ts = proc.decode(model.generate(**isyn)[0])
    from jiwer import wer  # local import for light start‑up

    return wer(tr, ts)
