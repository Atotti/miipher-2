import argparse

import webdataset as wds


def main(path: str) -> None:
    ds = wds.WebDataset(
        path,
    )
    data = next(iter(ds))
    print(data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path used in wds.WebDataset()")
    args = ap.parse_args()
    main(args.data)
