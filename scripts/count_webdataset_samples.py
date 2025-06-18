import argparse

import webdataset as wds


def main() -> None:
    parser = argparse.ArgumentParser(description="Count samples in a WebDataset")
    parser.add_argument(
        "path_pattern",
        type=str,
        help="Path pattern for the WebDataset shards (e.g., 'data/jvs-train-{000000..000020}.tar.gz')",
    )
    args = parser.parse_args()

    # resampled=False にして、1周でループが終了するように設定
    dataset = wds.WebDataset(args.path_pattern, resampled=False)

    count = 0
    for _ in dataset:
        count += 1
        if count % 1000 == 0:
            print(f"Counted {count} samples...")

    print(f"Total samples found: {count}")


if __name__ == "__main__":
    main()
