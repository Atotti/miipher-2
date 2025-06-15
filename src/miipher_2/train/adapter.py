import pathlib

import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader

from miipher_2.data.webdataset_loader import AdapterDataset
from miipher_2.model.feature_cleaner import FeatureCleaner


def train_adapter(cfg: DictConfig) -> None:
    # ---------------- Data ----------------
    dl = DataLoader(
        # 使用するクラスをAdapterDatasetに変更
        AdapterDataset(cfg.dataset.path_pattern, shuffle=cfg.dataset.shuffle),
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
    )
    # ---------------- Model ----------------
    model = FeatureCleaner().cuda()
    opt = optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=tuple(cfg.optim.betas),
    )
    scaler = torch.cuda.amp.GradScaler()
    l1 = nn.L1Loss()

    # ---------------- Train loop ----------------
    it = 0
    for ep in range(cfg.epochs):
        for noisy, clean in dl:
            noisy, clean = noisy.cuda(), clean.cuda()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model(noisy)
                target = model.extractor(clean)
                loss = l1(pred, target) + (pred - target).square().mean()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            if it % cfg.log_interval == 0:
                print(f"[Adapter] ep{ep + 1} it{it:>6}  L={loss.item():.4f}")
            it += 1

    # ---------------- Save ----------------
    sd = pathlib.Path(cfg.save_dir)
    sd.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), sd / "adapter_final.pt")
