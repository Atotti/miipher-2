import argparse
import pathlib

import torch

from miipher_2.hifigan.generator import Generator
from miipher_2.model.feature_cleaner import FeatureCleaner
from miipher_2.utils.audio import load, save


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=pathlib.Path, required=True)
    ap.add_argument("--vocoder", type=pathlib.Path, required=True)
    ap.add_argument("--in", dest="wav_in", type=pathlib.Path, required=True)
    ap.add_argument("--out", dest="wav_out", type=pathlib.Path, required=True)
    ap.add_argument("--hubert-layer", type=int, default=9, help="HuBERT layer to use (0-based)")
    return ap.parse_args()


def main() -> None:
    a = parse()
    cleaner = FeatureCleaner(hubert_layer=a.hubert_layer).cuda().eval()
    cleaner.load_state_dict(torch.load(a.adapter, map_location="cpu"))
    gen = Generator().cuda().eval()
    gen.load_state_dict(torch.load(a.vocoder, map_location="cpu")["gen"])
    wav = load(a.wav_in).cuda()  # 入力は16kHzにリサンプル
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        feat = cleaner(wav)
        restored = gen(feat)  # 出力は22.05kHz
    a.wav_out.parent.mkdir(parents=True, exist_ok=True)
    # 保存時にサンプリングレートを22050に指定
    save(a.wav_out, restored.cpu(), sr=22050)


if __name__ == "__main__":
    main()
