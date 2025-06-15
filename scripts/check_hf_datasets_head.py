import argparse

import datasets


def main(path: str, n_head: int) -> None:
    ds = datasets.load_from_disk(path)

    first_examples = ds.take(n_head)
    for i, ex in enumerate(first_examples, 1):
        print(f"[{i}] {ex}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path used in datasets.load_from_disk()")
    ap.add_argument("--head", type=int, default=10, help="show this many examples and exit (default: 10)")
    args = ap.parse_args()
    main(args.data, args.head)
