import io
import subprocess
from pathlib import Path

import torch
import torchaudio

SR = 16000


def load(path: str | Path) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.mean(0, keepdim=True)  # mono


def save(path: str | Path, wav: torch.Tensor, sr: int) -> None:
    torchaudio.save(path, wav, sr)


# --- 劣化関数 -----------------------------------------------
def add_noise(wav: torch.Tensor, snr_db: float = 20) -> torch.Tensor:
    noise = torch.randn_like(wav)
    sig_pow = wav.pow(2).mean()
    noise_pow = noise.pow(2).mean()
    factor = (sig_pow / noise_pow / (10 ** (snr_db / 10))) ** 0.5
    return wav + factor * noise


def add_reverb(wav: torch.Tensor, rt60: float = 0.3) -> torch.Tensor:
    # sox コンボリューションを手軽に呼ぶ
    cmd = ["sox", "-t", "wav", "-", "-t", "wav", "-", "reverb", f"{rt60 * 1000:.1f}"]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as p:  # noqa: S603
        out, _ = p.communicate(wav.cpu().numpy().astype("float32").tobytes())
    wav_r, _ = torchaudio.load(io.BytesIO(out))
    return wav_r.to(wav.device)


def codec(wav: torch.Tensor, codec_name: str = "mp3") -> torch.Tensor:
    if codec_name == "mp3":
        enc = ["-ar", "16000", "-ac", "1", "-codec:a", "libmp3lame", "-b:a", "64k"]
    elif codec_name == "opus":
        enc = ["-ar", "16000", "-ac", "1", "-codec:a", "libopus", "-b:a", "32k"]
    elif codec_name == "vorbis":
        enc = ["-codec:a", "libvorbis", "-qscale:a", "3"]
    elif codec_name == "alaw":
        enc = ["-codec:a", "alaw"]
    else:  # amr
        enc = ["-codec:a", "libopencore_amrwb", "-b:a", "16k"]
    cmd_enc = ["ffmpeg", "-f", "wav", "-i", "-", *enc, "-f", "wav", "-"]
    proc = subprocess.run(  # noqa: S603
        cmd_enc, input=wav.cpu().numpy().astype("float32").tobytes(), capture_output=True, check=False
    )
    wav_d, _ = torchaudio.load(io.BytesIO(proc.stdout))
    return wav_d.to(wav.device)
