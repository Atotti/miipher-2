import math
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


# バッチ内のテンソルの長さを揃える
def collate_tensors(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    noisy_tensors, clean_tensors = zip(*batch, strict=False)

    noisy_tensors = [x.squeeze(0) for x in noisy_tensors]  # (T)
    clean_tensors = [x.squeeze(0) for x in clean_tensors]  # (T)

    noisy = pad_sequence(noisy_tensors, batch_first=True)  # (B, Tmax)
    clean = pad_sequence(clean_tensors, batch_first=True)  # (B, Tmax)
    return noisy, clean


def train_adapter(cfg: DictConfig) -> None:
    # ---------------- チェックポイント確認と再開 ----------------
    resume_checkpoint_path = get_resume_checkpoint_path(cfg)
    resumed_checkpoint = None

    if resume_checkpoint_path:
        resumed_checkpoint = load_checkpoint(resume_checkpoint_path)
        restore_random_states(resumed_checkpoint)
        print(f"[INFO] Resuming from step {resumed_checkpoint['step']}")

    # ---------------- Wandb初期化 ----------------
    setup_wandb_resume(cfg, resumed_checkpoint)

    # ---------------- Data ----------------
    dl = DataLoader(
        AdapterDataset(cfg.dataset.path_pattern, shuffle=cfg.dataset.shuffle),
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        collate_fn=collate_tensors,
    )

    # ---------------- Model ----------------
    model = FeatureCleaner(cfg.model).cuda().float()

    # ターゲット生成用のクリーンなモデルを別途初期化
    target_model = (
        HubertExtractor(
            model_name=cfg.model.hubert_model_name,
            layer=cfg.model.hubert_layer,
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

    num_training_steps = math.ceil(cfg.dataset.num_examples / cfg.batch_size) * cfg.epochs

    scheduler = get_scheduler(
        name=cfg.optim.scheduler.name,
        optimizer=opt,
        num_warmup_steps=cfg.optim.scheduler.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # ---------------- チェックポイントから状態復元 ----------------
    start_epoch = 0
    start_it = 0

    if resumed_checkpoint:
        model.load_state_dict(resumed_checkpoint["model_state_dict"])
        opt.load_state_dict(resumed_checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in resumed_checkpoint:
            scheduler.load_state_dict(resumed_checkpoint["scheduler_state_dict"])
        start_it = resumed_checkpoint["step"]
        start_epoch = resumed_checkpoint.get("epoch", 0)
        print("[INFO] Restored model, optimizer, and scheduler states")

    mae_loss_fn = nn.L1Loss()
    mse_loss_fn = nn.MSELoss()

    # ---------------- Train loop ----------------
    it = start_it
    for ep in range(start_epoch, cfg.epochs):
        for noisy, clean in dl:
            # 再開時に既に処理済みのバッチをスキップ
            if it < start_it:
                it += 1
                continue

            noisy, clean = noisy.cuda(), clean.cuda()  # noqa: PLW2901
            pred = model(noisy)
            with torch.no_grad():
                target = target_model(clean)

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

            # ---------------- ログ出力 ----------------
            if it % cfg.log_interval == 0:
                log_data = {
                    "epoch": ep + 1,
                    "iteration": it,
                    "loss/total": loss.item(),
                    "loss/mae": mae_loss.item(),
                    "loss/mse": mse_loss.item(),
                    "loss/sc": sc_loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
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

            # ---------------- チェックポイント保存 ----------------
            if hasattr(cfg, "checkpoint") and it > 0 and it % cfg.checkpoint.save_interval == 0:
                save_checkpoint(
                    checkpoint_dir=cfg.save_dir,
                    step=it,
                    model_state=model.state_dict(),
                    optimizer_state=opt.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    epoch=ep,
                    cfg=cfg,
                    keep_last_n=cfg.checkpoint.keep_last_n,
                )

            it += 1

    # ---------------- 最終モデル保存 ----------------
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
