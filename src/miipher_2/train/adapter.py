import pathlib

import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader

from miipher_2.data.webdataset_loader import WavPairDataset
from miipher_2.model.feature_cleaner import FeatureCleaner


def train_adapter(cfg: DictConfig) -> None:  # hydra から直接呼ぶ
    # -------- dataset (WebDataset, already degraded) ----------
    ds = WavPairDataset(
        pattern=cfg.dataset.path_pattern,
        shuffle=cfg.dataset.shuffle,
    )
    dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=8, pin_memory=True)

    # -------- model ----------
    model = FeatureCleaner().cuda()
    opt = optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    l1 = nn.L1Loss()

    # -------- train ----------
    for ep in range(cfg.epochs):
        for noisy, clean in dl:
            noisy, clean = noisy.cuda(), clean.cuda()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model(noisy)  # (B,768,T)
                target = model.extractor(clean)  # Clean feature
                loss = l1(pred, target) + (pred - target).square().mean()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        print(f"[Adapter] epoch {ep + 1}/{cfg.epochs}  L={loss.item():.4f}")

    # -------- save ----------
    sd = pathlib.Path(cfg.save_dir)
    sd.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), sd / "adapter_final.pt")
