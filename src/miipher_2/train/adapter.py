# ------------------------------------------------------------
#  miipher/train/adapter.py
# ------------------------------------------------------------
"""
Parallel Adapter 単体学習スクリプト
"""

import argparse
import pathlib

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from miipher_2.data.dataloader import CleanNoisyDataset
from miipher_2.model.feature_cleaner import FeatureCleaner


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_list", type=pathlib.Path, required=True)
    ap.add_argument("--batch_sec", type=float, default=6)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save", type=pathlib.Path, default=pathlib.Path("exp/adapter"))
    return ap.parse_args()


def main() -> None:
    args = parse()
    wavs = [path.strip() for path in args.wav_list.read_text().splitlines()]
    ds = CleanNoisyDataset([pathlib.Path(p) for p in wavs])
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    model = FeatureCleaner().cuda()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler()
    l1 = nn.L1Loss()
    for ep in range(args.epochs):
        for noisy, clean in dl:
            noisy, clean = noisy.cuda(), clean.cuda()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model(noisy)  # (B,768,T)
                target = model.extractor(clean)  # clean feature
                loss = l1(pred, target) + (pred - target).pow(2).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
        print(f"epoch{ep}: {loss.item():.4f}")
    args.save.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save / "adapter_final.pt")


if __name__ == "__main__":
    main()
