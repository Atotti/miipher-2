import argparse
import pathlib

from miipher_2.utils.infer import main as infer_main


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=pathlib.Path, required=True)
    ap.add_argument("--vocoder", type=pathlib.Path, required=True)
    ap.add_argument("--in", dest="wav_in", type=pathlib.Path, required=True)
    ap.add_argument("--out", dest="wav_out", type=pathlib.Path, required=True)
    ap.add_argument("--hubert-layer", type=int, default=9, help="HuBERT layer to use (0-based)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse()
    # Pass arguments to infer_main
    import sys

    sys.argv = [
        sys.argv[0],
        "--adapter",
        str(args.adapter),
        "--vocoder",
        str(args.vocoder),
        "--in",
        str(args.wav_in),
        "--out",
        str(args.wav_out),
        "--hubert-layer",
        str(args.hubert_layer),
    ]
    infer_main()
