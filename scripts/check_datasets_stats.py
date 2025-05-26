import argparse
import collections

import datasets
import tqdm

SR = 24_000


def main(path: str) -> None:
    ds = datasets.load_from_disk(path).cast_column("audio", datasets.Audio(sampling_rate=SR))
    utterances = len(ds)
    langs = collections.defaultdict(float)
    tot_samples = 0

    for ex in tqdm.tqdm(ds, desc="scanning"):
        n = len(ex["audio"]["array"])
        tot_samples += n
        langs[ex["language"]] += n / SR

    tot_hours = tot_samples / SR / 3600

    print("=== Dataset Stats ===")
    print(f" utterances         : {utterances:,}")
    print(f" total duration     : {tot_hours:,.2f} h")
    print(" duration by language (h):")
    for lg, sec in sorted(langs.items(), key=lambda x: -x[1]):
        print(f"  {lg:<5} : {sec / 3600:6.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path passed to datasets.load_from_disk()")
    main(ap.parse_args().data)
