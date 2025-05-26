import argparse
import csv
import pathlib
import subprocess
from collections.abc import Iterator
from typing import Any

import datasets
import tqdm

SR = 24_000


def wav_to_flac(root: pathlib.Path, out_root: pathlib.Path) -> None:
    wav_files = list(root.rglob("*.wav"))
    print(f"Converting {len(wav_files):,} WAV ⇒ FLAC …")
    with tqdm.tqdm(total=len(wav_files), unit="wav") as progress_bar:
        for wav in wav_files:
            rel = wav.relative_to(root)
            flac = out_root / rel.with_suffix(".flac")
            if flac.exists():
                progress_bar.update(1)
                continue
            flac.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(  # noqa: S603
                ["sox", wav, "-C", "0", flac],  # noqa: S607
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            progress_bar.update(1)
    print(" FLAC 変換完了")


def iter_trans_tsv(root: pathlib.Path, flac_root: pathlib.Path) -> Iterator[tuple[pathlib.Path, str]]:
    for tsv in root.rglob("*.trans.tsv"):
        with tsv.open("r", encoding="utf-8") as f:
            reader = csv.reader(
                f,
                delimiter="\t",
                quoting=csv.QUOTE_NONE,
                escapechar="\\",
            )
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                utt_id, *_, norm_txt = row
                flac_path = flac_root / tsv.parent.relative_to(root) / f"{utt_id}.flac"
                if flac_path.exists():
                    yield flac_path.relative_to(flac_root), norm_txt


def gen_examples(root: pathlib.Path, flac_root: pathlib.Path) -> Iterator[dict[str, Any]]:
    for rel_flac, text in tqdm.tqdm(iter_trans_tsv(root, flac_root), desc="yield"):
        abs_flac = flac_root / rel_flac
        yield {
            "audio": str(abs_flac),
            "text": text,
            "speaker_id": "",
            "language": "en",
            "source": "libritts_r",
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LibriTTS-R の WAV ルート")
    ap.add_argument("--out_dir", required=True, help="save_to_disk() の出力フォルダ")
    ap.add_argument("--dataset_name", required=True, help="データセットの名前")
    args = ap.parse_args()

    root = pathlib.Path(args.root).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    dataset_name = str(args.dataset_name)
    flac_root = out_dir / dataset_name / "flac"
    flac_root.mkdir(parents=True, exist_ok=True)

    # wav_to_flac(root, flac_root)

    features = datasets.Features(
        {
            "audio": datasets.Audio(sampling_rate=SR, decode=False),  # decode=False なのでパスだけ保持
            "text": datasets.Value("string"),
            "speaker_id": datasets.Value("string"),
            "language": datasets.Value("string"),
            "source": datasets.Value("string"),
        }
    )
    ds = datasets.Dataset.from_generator(
        lambda: gen_examples(root, flac_root),
        features=features,
    )
    ds.save_to_disk(out_dir / dataset_name)
    print("✅  saved  =>", out_dir / dataset_name)


if __name__ == "__main__":
    main()
