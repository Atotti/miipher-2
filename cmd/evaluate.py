import argparse
import glob
import pathlib

import dnsmos_pytorch
import numpy as np
import squid_py
import torch
import tqdm
from speaker_verification_toolkit import get_embedding
from whisperx import load_model

from miipher_2.utils.audio import load


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
        cw, rw = load(c), load(r)
        sig.append(dns_model(rw)["SIG"])
        squid.append(squid_py.score(cw.squeeze(0), rw.squeeze(0)))
        wer.append(whisper(rw.squeeze(0))["wer"])
        spk.append(torch.cosine_similarity(get_embedding(cw), get_embedding(rw), 0).item())
    print(f"DNSMOS‑SIG {np.mean(sig):.3f}")
    print(f"SQuId      {np.mean(squid):.3f}")
    print(f"WER        {np.mean(wer) * 100:.2f}%")
    print(f"SPK cos    {np.mean(spk):.3f}")


if __name__ == "__main__":
    main()
