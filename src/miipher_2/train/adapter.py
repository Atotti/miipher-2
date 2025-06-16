import pathlib

import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import wandb
from miipher_2.data.webdataset_loader import AdapterDataset
from miipher_2.model.feature_cleaner import FeatureCleaner


# バッチ内のテンソルの長さを揃える
def collate_tensors(batch):
    noisy_tensors, clean_tensors = zip(*batch, strict=False)

    noisy_tensors = [x.squeeze(0) for x in noisy_tensors]  # (T)
    clean_tensors = [x.squeeze(0) for x in clean_tensors]  # (T)

    noisy = pad_sequence(noisy_tensors, batch_first=True)  # (B, Tmax)
    clean = pad_sequence(clean_tensors, batch_first=True)  # (B, Tmax)
    return noisy, clean


def train_adapter(cfg: DictConfig) -> None:
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes,
            config=dict(cfg),
        )

    # ---------------- Data ----------------
    dl = DataLoader(
        AdapterDataset(cfg.dataset.path_pattern, shuffle=cfg.dataset.shuffle),
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        collate_fn=collate_tensors,
    )
    # ---------------- Model ----------------
    model = FeatureCleaner(cfg.model).cuda()
    opt = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=tuple(cfg.optim.betas),
    )
    scaler = torch.amp.GradScaler("cuda")

    mae_loss_fn = nn.L1Loss()
    mse_loss_fn = nn.MSELoss()

    # ---------------- Train loop ----------------
    it = 0
    for ep in range(cfg.epochs):
        for noisy, clean in dl:
            noisy, clean = noisy.cuda(), clean.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(noisy)
                with torch.no_grad():
                    target = model.extractor(clean)

                min_len = min(pred.size(2), target.size(2))
                pred = pred[:, :, :min_len]
                target = target[:, :, :min_len]

                mae_loss = mae_loss_fn(pred, target)
                mse_loss = mse_loss_fn(pred, target)
                sc_loss = (pred - target).pow(2).sum() / (target.pow(2).sum() + 1e-9)
                loss = mae_loss + mse_loss + sc_loss

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.optim.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            if it % cfg.log_interval == 0:
                log_data = {
                    "epoch": ep + 1,
                    "iteration": it,
                    "loss/total": loss.item(),
                    "loss/mae": mae_loss.item(),
                    "loss/mse": mse_loss.item(),
                    "loss/sc": sc_loss.item(),
                    "learning_rate": opt.param_groups[0]["lr"],
                }

                print(
                    f"[Adapter] ep{ep + 1} it{it:>6} | "
                    f"Total Loss={loss.item():.4f} | "
                    f"MAE={mae_loss.item():.4f} | "
                    f"MSE={mse_loss.item():.4f} | "
                    f"SC={sc_loss.item():.4f}"
                )

                if cfg.wandb.enabled:
                    wandb.log(log_data, step=it)
            it += 1

    # ---------------- Save ----------------
    sd = pathlib.Path(cfg.save_dir)
    sd.mkdir(parents=True, exist_ok=True)
    model_path = sd / "adapter_final.pt"
    torch.save(model.state_dict(), model_path)

    # Log model as wandb artifact
    if cfg.wandb.enabled and cfg.wandb.log_model:
        artifact = wandb.Artifact("adapter_model", type="model")
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)
        print("[INFO] Model saved as wandb artifact")

    if cfg.wandb.enabled:
        wandb.finish()
