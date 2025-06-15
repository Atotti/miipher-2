import pathlib

import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader

from miipher_2.data.dataloader import CleanNoisyDataset
from miipher_2.model.feature_cleaner import FeatureCleaner


def train_adapter(cfg: DictConfig) -> None:
    wav_list: pathlib.Path = pathlib.Path(cfg.dataset.list)

    # ---------- データセット ----------
    wavs = [p.strip() for p in wav_list.read_text().splitlines()]
    ds = CleanNoisyDataset([pathlib.Path(p) for p in wavs])
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    model = FeatureCleaner().cuda()
    opt = optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    l1 = nn.L1Loss()
    for ep in range(cfg.epochs):
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
    save_dir = pathlib.Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "adapter_final.pt")
