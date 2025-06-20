import pathlib
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import get_scheduler

from miipher_2.accelerate_utils import build_accelerator, log_metrics, print_main, setup_random_seeds
from miipher_2.data.webdataset_loader import AdapterDataset
from miipher_2.extractors.hubert import HubertExtractor
from miipher_2.model.feature_cleaner import FeatureCleaner


def collate_tensors(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    noisy_tensors, clean_tensors = zip(*batch, strict=False)
    noisy_tensors = [x.squeeze(0) for x in noisy_tensors]
    clean_tensors = [x.squeeze(0) for x in clean_tensors]
    noisy = pad_sequence(noisy_tensors, batch_first=True, padding_value=0.0)
    clean = pad_sequence(clean_tensors, batch_first=True, padding_value=0.0)
    return noisy, clean


@torch.no_grad()
def validate(
    model: nn.Module,
    target_model: nn.Module,
    val_dl: DataLoader,
    loss_fns: dict[str, nn.Module],
    accelerator: Any,
    cfg: DictConfig,
) -> dict[str, float]:
    """
    Adapter検証関数（Accelerate対応版）
    """
    model.eval()
    total_losses = {"total": 0.0, "mae": 0.0, "mse": 0.0, "sc": 0.0}
    total_count = 0

    limit_batches = cfg.get("validation_batches", None)

    for i, (noisy, clean) in enumerate(val_dl):
        if limit_batches is not None and i >= limit_batches:
            break

        with accelerator.autocast():
            target = target_model(clean)
            pred = model(noisy)

            min_len = min(pred.size(2), target.size(2))
            pred, target = pred[:, :, :min_len], target[:, :, :min_len]

            mae_loss = loss_fns["mae"](pred, target)
            mse_loss = loss_fns["mse"](pred, target)
            sc_loss = (pred - target).pow(2).sum() / (target.pow(2).sum() + 1e-9)
            loss = mae_loss + mse_loss + sc_loss

        # Accelerateのgatherを使用してマルチGPUでの損失を集約
        all_losses = accelerator.gather(torch.stack([loss, mae_loss, mse_loss, sc_loss]))
        batch_sizes = accelerator.gather(torch.tensor(noisy.size(0), device=accelerator.device))

        total_losses["total"] += all_losses[0].sum().item()
        total_losses["mae"] += all_losses[1].sum().item()
        total_losses["mse"] += all_losses[2].sum().item()
        total_losses["sc"] += all_losses[3].sum().item()
        total_count += batch_sizes.sum().item()

    avg_losses = {f"val_loss/{key}": val / total_count for key, val in total_losses.items()}
    model.train()
    return avg_losses


def train_adapter(cfg: DictConfig) -> None:
    """
    Accelerateベースの並列Adapter学習
    """
    # Acceleratorとロガーの初期化
    accelerator, logger = build_accelerator(cfg)

    # 乱数シードの設定
    setup_random_seeds(accelerator, cfg.get("seed", 42))

    # データローダーの準備
    train_ds = AdapterDataset(cfg.dataset.path_pattern, shuffle=cfg.dataset.shuffle)
    val_ds = AdapterDataset(cfg.dataset.val_path_pattern, shuffle=False)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        collate_fn=collate_tensors,
        drop_last=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        collate_fn=collate_tensors,
        drop_last=False,
    )

    # モデルの準備
    model = FeatureCleaner(cfg.model).float()
    target_model = HubertExtractor(
        model_name=cfg.model.hubert_model_name,
        layer=cfg.model.hubert_layer,
    ).float().eval()

    # target_modelの勾配を無効化
    for param in target_model.parameters():
        param.requires_grad = False

    # オプティマイザーとスケジューラーの準備
    opt = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=tuple(cfg.optim.betas),
    )

    scheduler = get_scheduler(
        name=cfg.optim.scheduler.name,
        optimizer=opt,
        num_warmup_steps=cfg.optim.scheduler.warmup_steps,
        num_training_steps=cfg.steps,
    )

    # Accelerator.prepareで一括準備
    model, target_model, opt, scheduler, train_dl, val_dl = accelerator.prepare(
        model, target_model, opt, scheduler, train_dl, val_dl
    )

    # 損失関数の準備
    mae_loss_fn = nn.L1Loss()
    mse_loss_fn = nn.MSELoss()
    loss_fns = {"mae": mae_loss_fn, "mse": mse_loss_fn}

    # チェックポイントからの再開
    start_step = 0
    checkpoint_dir = pathlib.Path(cfg.save_dir)
    if (checkpoint_dir / "pytorch_model.bin").exists():
        print_main(accelerator, f"Loading checkpoint from {checkpoint_dir}")
        accelerator.load_state(checkpoint_dir)
        # ステップ数の復元（必要に応じてファイルから読み込み）
        step_file = checkpoint_dir / "step.txt"
        if step_file.exists():
            start_step = int(step_file.read_text()) + 1
            print_main(accelerator, f"Resuming from step {start_step}")

    # データローダーのイテレーター作成
    train_iter = iter(train_dl)

    print_main(accelerator, "Starting Adapter training with Accelerate")
    print_main(accelerator, f"Mixed precision: {accelerator.mixed_precision}")
    print_main(accelerator, f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")

    # 学習ループ
    for step in range(start_step, cfg.steps):
        # 検証の実行
        if step > 0 and step % cfg.validation_interval == 0:
            val_losses = validate(
                model, target_model, val_dl, loss_fns, accelerator, cfg
            )
            log_metrics(accelerator, val_losses, step)
            print_main(accelerator, f"[Adapter] Step:{step:>7} | Validation Loss: {val_losses['val_loss/total']:.4f}")

        # データの取得（イテレーター循環処理）
        try:
            noisy, clean = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            noisy, clean = next(train_iter)

        # 勾配累積のコンテキスト内で学習
        with accelerator.accumulate(model):
            with accelerator.autocast():
                # ターゲット特徴量の計算（勾配なし）
                with torch.no_grad():
                    target = target_model(clean)

                # 予測の計算
                pred = model(noisy)

                # 長さの調整
                min_len = min(pred.size(2), target.size(2))
                pred = pred[:, :, :min_len]
                target = target[:, :, :min_len]

                # 損失の計算
                mae_loss = mae_loss_fn(pred, target)
                mse_loss = mse_loss_fn(pred, target)
                sc_loss = (pred - target).pow(2).sum() / (target.pow(2).sum() + 1e-9)
                loss = mae_loss + mse_loss + sc_loss

            # 逆伝播と最適化
            accelerator.backward(loss)

            # 勾配クリッピング
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=cfg.optim.max_grad_norm)

            opt.step()
            scheduler.step()
            opt.zero_grad()

        # ログ出力
        if step % cfg.log_interval == 0:
            metrics = {
                "train/loss_total": loss.item(),
                "train/loss_mae": mae_loss.item(),
                "train/loss_mse": mse_loss.item(),
                "train/loss_sc": sc_loss.item(),
                "lr": scheduler.get_last_lr()[0],
            }
            log_metrics(accelerator, metrics, step)
            print_main(
                accelerator,
                f"[Adapter] Step:{step:>7} | "
                f"Total Loss={loss.item():.4f} | "
                f"MAE={mae_loss.item():.4f} | "
                f"MSE={mse_loss.item():.4f} | "
                f"SC={sc_loss.item():.4f}",
            )

        # チェックポイントの保存
        if step > 0 and step % cfg.checkpoint.save_interval == 0:
            print_main(accelerator, f"Saving checkpoint at step {step}")
            accelerator.save_state(output_dir=cfg.save_dir, safe_serialization=False)
            # ステップ数も保存
            if accelerator.is_main_process:
                (pathlib.Path(cfg.save_dir) / "step.txt").write_text(str(step))

    # 学習終了時の処理
    print_main(accelerator, "Training completed")
    if accelerator.is_main_process:
        accelerator.save_state(output_dir=cfg.save_dir, safe_serialization=False)
        accelerator.end_training()
