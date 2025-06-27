import pathlib

import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import get_scheduler

import wandb
from miipher_2.data.webdataset_loader import AdapterDataset
from miipher_2.extractors.hubert import HubertExtractor
from miipher_2.model.feature_cleaner import FeatureCleaner
from miipher_2.utils.checkpoint import (
    get_resume_checkpoint_path,
    load_checkpoint,
    restore_random_states,
    save_checkpoint,
    setup_wandb_resume,
)


def collate_tensors(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    noisy_tensors, clean_tensors = zip(*batch, strict=False)
    noisy_tensors = [x.squeeze(0) for x in noisy_tensors]
    clean_tensors = [x.squeeze(0) for x in clean_tensors]
    noisy = pad_sequence(noisy_tensors, batch_first=True, padding_value=0.0)
    clean = pad_sequence(clean_tensors, batch_first=True, padding_value=0.0)
    return noisy, clean


@torch.no_grad()
def validate(
    model: nn.Module, target_model: nn.Module, val_dl: DataLoader, loss_fns: dict, limit_batches: int | None = None
) -> dict:
    model.eval()
    total_losses = {"total": 0.0, "mae": 0.0, "mse": 0.0, "sc": 0.0}
    total_count = 0

    for i, (noisy, clean) in enumerate(val_dl):
        if limit_batches is not None and i >= limit_batches:
            break
        noisy, clean = noisy.cuda(), clean.cuda()  # noqa: PLW2901

        target = target_model(clean)
        pred = model(noisy)

        min_len = min(pred.size(2), target.size(2))
        pred, target = pred[:, :, :min_len], target[:, :, :min_len]

        mae_loss = loss_fns["mae"](pred, target)
        mse_loss = loss_fns["mse"](pred, target)
        sc_loss = (pred - target).pow(2).sum() / (target.pow(2).sum() + 1e-9)
        loss = mae_loss + mse_loss + sc_loss

        total_losses["total"] += loss.item() * noisy.size(0)
        total_losses["mae"] += mae_loss.item() * noisy.size(0)
        total_losses["mse"] += mse_loss.item() * noisy.size(0)
        total_losses["sc"] += sc_loss.item() * noisy.size(0)
        total_count += noisy.size(0)

    avg_losses = {key: val / total_count for key, val in total_losses.items()}
    model.train()
    return avg_losses


def train_adapter(cfg: DictConfig) -> None:
    resume_checkpoint_path = get_resume_checkpoint_path(cfg)
    resumed_checkpoint = None
    if resume_checkpoint_path:
        resumed_checkpoint = load_checkpoint(str(resume_checkpoint_path))
        restore_random_states(resumed_checkpoint)
        print(f"[INFO] Resuming from step {resumed_checkpoint['step']}")

    setup_wandb_resume(cfg, resumed_checkpoint)

    dl = DataLoader(
        AdapterDataset(cfg.dataset.path_pattern, shuffle=cfg.dataset.shuffle),
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        collate_fn=collate_tensors,
        drop_last=True,
    )

    val_dl = DataLoader(
        AdapterDataset(cfg.dataset.val_path_pattern, shuffle=False),  # シャッフルは不要
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        collate_fn=collate_tensors,
        drop_last=False,
    )

    model = FeatureCleaner(cfg.model).cuda().float()

    target_model = (
        HubertExtractor(
            model_name=cfg.model.hubert_model_name,
            layer=cfg.model.hubert_layer - 1,
        )
        .cuda()
        .float()
        .eval()
    )
    for param in target_model.parameters():
        param.requires_grad = False

    opt = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=tuple(cfg.optim.betas),
    )

    # スケジューラの総ステップ数をcfg.stepsから直接取得
    scheduler = get_scheduler(
        name=cfg.optim.scheduler.name,
        optimizer=opt,
        num_warmup_steps=cfg.optim.scheduler.warmup_steps,
        num_training_steps=cfg.steps,
    )

    start_it = 0
    if resumed_checkpoint:
        model.load_state_dict(resumed_checkpoint["model_state_dict"])
        opt.load_state_dict(resumed_checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in resumed_checkpoint:
            scheduler.load_state_dict(resumed_checkpoint["scheduler_state_dict"])
        start_it = resumed_checkpoint.get("step", 0) + 1
        print("[INFO] Restored model, optimizer, and scheduler states")

    mae_loss_fn = nn.L1Loss()
    mse_loss_fn = nn.MSELoss()

    dl_iter = iter(dl)

    for it in range(start_it, cfg.steps):
        if it > 0 and it % cfg.validation_interval == 0:
            val_losses = validate(
                model,
                target_model,
                val_dl,
                {"mae": mae_loss_fn, "mse": mse_loss_fn},
                limit_batches=cfg.get("validation_batches"),
            )
            wandb.log({f"val_loss/{key}": val for key, val in val_losses.items()}, step=it)
            print(f"[Adapter] it:{it:>7} | Validation Loss: {val_losses['total']:.4f}")

        try:
            noisy, clean = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            noisy, clean = next(dl_iter)

        noisy, clean = noisy.cuda(), clean.cuda()

        with torch.no_grad():
            target = target_model(clean)

        pred = model(noisy)

        min_len = min(pred.size(2), target.size(2))
        pred = pred[:, :, :min_len]
        target = target[:, :, :min_len]

        mae_loss = mae_loss_fn(pred, target)
        mse_loss = mse_loss_fn(pred, target)
        sc_loss = (pred - target).pow(2).sum() / (target.pow(2).sum() + 1e-9)
        loss = mae_loss + mse_loss + sc_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.optim.max_grad_norm)
        opt.step()
        scheduler.step()

        if it % cfg.log_interval == 0:
            log_data = {
                "iteration": it,
                "loss/total": loss.item(),
                "loss/mae": mae_loss.item(),
                "loss/mse": mse_loss.item(),
                "loss/sc": sc_loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            }
            print(
                f"[Adapter] it:{it:>7} | "
                f"Total Loss={loss.item():.4f} | "
                f"MAE={mae_loss.item():.4f} | "
                f"MSE={mse_loss.item():.4f} | "
                f"SC={sc_loss.item():.4f}"
            )
            if cfg.wandb.enabled:
                wandb.log(log_data, step=it)

        if hasattr(cfg, "checkpoint") and it > 0 and it % cfg.checkpoint.save_interval == 0:
            save_checkpoint(
                checkpoint_dir=cfg.save_dir,
                step=it,
                model_state=model.state_dict(),
                optimizer_state=opt.state_dict(),
                scheduler_state=scheduler.state_dict(),
                cfg=cfg,
                keep_last_n=cfg.checkpoint.keep_last_n,
            )

    # 最終モデル保存
    sd = pathlib.Path(cfg.save_dir)
    sd.mkdir(parents=True, exist_ok=True)
    model_path = sd / "adapter_final.pt"
    torch.save(model.state_dict(), model_path)

    if cfg.wandb.enabled and cfg.wandb.log_model:
        artifact = wandb.Artifact("adapter_model", type="model")
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)
        print("[INFO] Model saved as wandb artifact")

    if cfg.wandb.enabled:
        wandb.finish()
