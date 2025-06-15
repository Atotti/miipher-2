import argparse
import pathlib

from miipher_2.infer import main as infer_main


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=pathlib.Path, required=True)
    ap.add_argument("--vocoder", type=pathlib.Path, required=True)
    ap.add_argument("--in", dest="wav_in", type=pathlib.Path, required=True)
    ap.add_argument("--out", dest="wav_out", type=pathlib.Path, required=True)
    return ap.parse_args()


if __name__ == "__main__":
    infer_main(parse())
