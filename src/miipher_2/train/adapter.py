import pathlib

import torch
from accelerate import Accelerator
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
    accelerator: Accelerator,
    model: nn.Module,
    target_model: nn.Module,
    val_dl: DataLoader,
    loss_fns: dict,
    limit_batches: int | None = None,
) -> dict:
    model.eval()
    total_losses = {"total": 0.0, "mae": 0.0, "mse": 0.0, "sc": 0.0}
    total_count = 0

    for i, (noisy, clean) in enumerate(val_dl):
        if limit_batches is not None and i >= limit_batches:
            break

        with torch.no_grad():
            target = target_model(clean)
        pred = model(noisy)

        min_len = min(pred.size(2), target.size(2))
        pred, target = pred[:, :, :min_len], target[:, :, :min_len]

        mae_loss = loss_fns["mae"](pred, target)
        mse_loss = loss_fns["mse"](pred, target)
        sc_loss = (pred - target).pow(2).sum() / (target.pow(2).sum() + 1e-9)
        loss = mae_loss + mse_loss + sc_loss

        mae_loss = accelerator.gather_for_metrics(mae_loss)
        mse_loss = accelerator.gather_for_metrics(mse_loss)
        sc_loss = accelerator.gather_for_metrics(sc_loss)
        loss = accelerator.gather_for_metrics(loss)

        total_losses["mae"] += mae_loss.sum().item()
        total_losses["mse"] += mse_loss.sum().item()
        total_losses["sc"] += sc_loss.sum().item()
        total_losses["total"] += loss.sum().item()

        total_count += noisy.size(0) * accelerator.num_processes

    avg_losses = {key: val / total_count for key, val in total_losses.items()}
    model.train()
    return avg_losses


def train_adapter(cfg: DictConfig) -> None:
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with="wandb",
    )
    resume_checkpoint_path = get_resume_checkpoint_path(cfg)
    resumed_checkpoint = None
    if resume_checkpoint_path:
        resumed_checkpoint = load_checkpoint(str(resume_checkpoint_path))
        restore_random_states(resumed_checkpoint)
        print(f"[INFO] Resuming from step {resumed_checkpoint['step']}")

    setup_wandb_resume(cfg, resumed_checkpoint)

    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        config=dict(cfg),
        init_kwargs={"entity": cfg.wandb.entity, "name": cfg.wandb.name, "tags": cfg.wandb.tags},
    )

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

    model = FeatureCleaner(cfg.model).float()

    target_model = (
        HubertExtractor(
            model_name=cfg.model.hubert_model_name,
            layer=cfg.model.hubert_layer,
        )
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

    model, opt, dl, val_dl, scheduler, target_model = accelerator.prepare(
        model, opt, dl, val_dl, scheduler, target_model
    )

    start_it = 0

    # --- チェックポイントからの再開 ---
    resume_checkpoint_path = get_resume_checkpoint_path(cfg)
    if resume_checkpoint_path and False:  # noqa: SIM223
        print(f"[INFO] Resuming from checkpoint: {resume_checkpoint_path}")
        # ステップ数を先に読み込む
        # (load_checkpointはCPUにロードするので、全プロセスで実行しても問題ない)
        resumed_checkpoint = load_checkpoint(str(resume_checkpoint_path))
        start_it = resumed_checkpoint.get("step", 0) + 1

        # 乱数状態を復元
        restore_random_states(resumed_checkpoint)

        # acceleratorがモデル・オプティマイザ等の状態をロードする
        accelerator.load_state(str(resume_checkpoint_path.parent))  # ディレクトリを渡す
        print(f"[INFO] Resumed from step {start_it}")

    # --- WandBトラッカーの初期化 ---
    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        config=dict(cfg),
        init_kwargs={"entity": cfg.wandb.entity, "name": cfg.wandb.name, "tags": cfg.wandb.tags},
    )

    mae_loss_fn = nn.L1Loss()
    mse_loss_fn = nn.MSELoss()

    dl_iter = iter(dl)

    for it in range(start_it, cfg.steps):
        if it > 0 and it % cfg.validation_interval == 0 and accelerator.is_main_process:
            # accelerator.unwrap_modelで元のモデルを取得
            unwrapped_model = accelerator.unwrap_model(model)
            val_losses = validate(
                accelerator,
                unwrapped_model,
                target_model,  # prepare済み
                val_dl,
                {"mae": mae_loss_fn, "mse": mse_loss_fn},
                limit_batches=cfg.get("validation_batches"),
            )
            accelerator.log({f"val_loss/{key}": val for key, val in val_losses.items()}, step=it)
            print(f"[Adapter] it:{it:>7} | Validation Loss: {val_losses['total']:.4f}")
        with accelerator.accumulate(model):
            try:
                noisy, clean = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                noisy, clean = next(dl_iter)

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

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=cfg.optim.max_grad_norm)

            opt.step()
            scheduler.step()
            opt.zero_grad()

        if it % cfg.log_interval == 0 and accelerator.is_main_process:
            log_data = {
                "iteration": it,
                "loss/total": loss.item(),
                "loss/mae": mae_loss.item(),
                "loss/mse": mse_loss.item(),
                "loss/sc": sc_loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            }
            accelerator.log(log_data, step=it)

        if hasattr(cfg, "checkpoint") and it > 0 and it % cfg.checkpoint.save_interval == 0:
            accelerator.save_state(output_dir=pathlib.Path(cfg.save_dir) / f"checkpoint_{it}k")

    # 最終モデル保存
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), pathlib.Path(cfg.save_dir) / "adapter_final.pt")

    accelerator.end_training()
