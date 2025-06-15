import argparse
import glob
import pathlib

import dnsmos_pytorch
import numpy as np
import squid_py
import torch
import torchaudio
import tqdm
from speaker_verification_toolkit import get_embedding
from whisperx import load_model


def load_for_eval(path, target_sr=22050):
    """評価用に音声をロード・リサンプリングする関数"""
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.mean(0, keepdim=True)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", type=pathlib.Path, required=True)
    ap.add_argument("--rest_dir", type=pathlib.Path, required=True)
    return ap.parse_args()


def main() -> None:
    a = parse()
    clean_files = sorted(glob.glob(str(a.clean_dir / "*.wav")))
    rest_files = [str(a.rest_dir / pathlib.Path(f).name) for f in clean_files]
    dns_model = dnsmos_pytorch.DNSMOS.from_pretrained()
    whisper = load_model("large-v3", device="cpu")
    sig, squid, wer, spk = [], [], [], []
    for c, r in tqdm.tqdm(list(zip(clean_files, rest_files, strict=False))):
        # 修正: 22.05kHzで読み込む
        cw_22k = load_for_eval(c, target_sr=22050)
        rw_22k = load_for_eval(r, target_sr=22050)

        # 評価モデルへの入力は16kHzが適切な場合があるため、必要に応じてリサンプリング
        rw_16k = torchaudio.functional.resample(rw_22k, 22050, 16000)

        # DNSMOS / Whisperは16kHz入力を想定していることが多い
        sig.append(dns_model(rw_16k, 16000)["SIG"])
        wer.append(whisper.transcribe(rw_16k.squeeze().numpy())["text"])  # whisperxの入力はnumpy array

        # SQuId / SPKは22.05kHzで比較
        squid.append(squid_py.score(cw_22k.squeeze(0), rw_22k.squeeze(0), 22050))
        spk.append(torch.cosine_similarity(get_embedding(cw_22k, 22050), get_embedding(rw_22k, 22050), 0).item())

    print(f"DNSMOS‑SIG {np.mean(sig):.3f}")
    print(f"SQuId      {np.mean(squid):.3f}")
    print(f"WER        {np.mean(wer) * 100:.2f}%")
    print(f"SPK cos    {np.mean(spk):.3f}")


if __name__ == "__main__":
    main()
