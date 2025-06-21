import itertools
import pathlib
from functools import partial
from typing import Any

import torch
from omegaconf import DictConfig
from torch import Tensor, nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from miipher_2.accelerate_utils import build_accelerator, log_metrics, print_main, setup_random_seeds, worker_init_fn
from miipher_2.data.webdataset_loader import AdapterDataset
from miipher_2.extractors.hubert import HubertExtractor
from miipher_2.model.feature_cleaner import FeatureCleaner


def collate_tensors(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """バッチ内のテンソルをパディングして統一"""
    noisy_list, clean_list = zip(*batch, strict=False)
    # pad_sequenceを使用してバッチ次元でパディング
    noisy_batch = pad_sequence(noisy_list, batch_first=True)  # (B, max_len)
    clean_batch = pad_sequence(clean_list, batch_first=True)  # (B, max_len)
    return noisy_batch, clean_batch


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
    検証の実行（分散対応 - パディングによるクラッシュ回避）
    """
    model.eval()
    total_losses = {"total": 0.0, "mae": 0.0, "mse": 0.0, "sc": 0.0}
    total_count = 0

    for noisy, clean in val_dl:
        with accelerator.autocast():
            # ターゲット特徴量の計算（勾配なし）
            target = target_model(clean)

            # 予測の計算
            pred = model(noisy)

            # 長さの調整
            min_len = min(pred.size(2), target.size(2))
            pred = pred[:, :, :min_len]
            target = target[:, :, :min_len]

            # 損失の計算（学習ループと同じ正規化）
            mae_loss = loss_fns["mae"](pred, target)
            mse_loss = loss_fns["mse"](pred, target)

            # ★★★ Spectral Convergence Loss: 学習ループと同じ正規化方式
            pred_power = pred.pow(2).mean(dim=(1, 2))  # (B,) - バッチ毎の平均パワー
            target_power = target.pow(2).mean(dim=(1, 2))  # (B,) - バッチ毎の平均パワー
            diff_power = (pred - target).pow(2).mean(dim=(1, 2))  # (B,) - バッチ毎の差分パワー
            sc_loss = (diff_power / (target_power + 1e-9)).mean()  # バッチ全体の平均

            loss = mae_loss + mse_loss + sc_loss

        # ★★★ 分散環境での安全な損失集約（パディングによるクラッシュ回避）
        loss_tensor = torch.stack([loss, mae_loss, mse_loss, sc_loss])
        batch_size_tensor = torch.tensor(noisy.size(0), device=accelerator.device, dtype=torch.float32)

        # プロセス間でテンソルサイズを揃える（最終バッチ対策）
        loss_tensor_padded = accelerator.pad_across_processes(loss_tensor, dim=0, pad_index=0.0)
        batch_size_padded = accelerator.pad_across_processes(batch_size_tensor, dim=0, pad_index=0.0)

        # 安全にgatherで集約
        all_losses = accelerator.gather(loss_tensor_padded)
        batch_sizes = accelerator.gather(batch_size_padded)

        # メインプロセスでのみ蓄積（重複回避）
        if accelerator.is_main_process:
            total_losses["total"] += all_losses[0].sum().item()
            total_losses["mae"] += all_losses[1].sum().item()
            total_losses["mse"] += all_losses[2].sum().item()
            total_losses["sc"] += all_losses[3].sum().item()
            total_count += batch_sizes.sum().item()

    # 平均損失計算（メインプロセスのみ）
    if accelerator.is_main_process and total_count > 0:
        avg_losses = {f"val_loss/{key}": val / total_count for key, val in total_losses.items()}
    else:
        avg_losses = {}

    # すべてのプロセスに結果をブロードキャスト（同期確保）
    avg_losses = accelerator.gather_for_metrics(avg_losses) if avg_losses else {}

    model.train()
    return avg_losses


def train_adapter(cfg: DictConfig) -> None:
    """
    Accelerateベースの並列Adapter学習（統一チェックポイント管理）
    """
    # Acceleratorとロガーの初期化
    accelerator, logger = build_accelerator(cfg)

    # 乱数シードの設定
    setup_random_seeds(accelerator, cfg.get("seed", 42))

    # データローダーの準備（再現性向上）
    train_ds = AdapterDataset(cfg.dataset.path_pattern, shuffle=cfg.dataset.shuffle)
    val_ds = AdapterDataset(cfg.dataset.val_path_pattern, shuffle=False)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        collate_fn=collate_tensors,
        drop_last=True,
        worker_init_fn=partial(worker_init_fn, seed=cfg.get("seed", 42)),  # 再現性向上
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        collate_fn=collate_tensors,
        drop_last=False,
        worker_init_fn=partial(worker_init_fn, seed=cfg.get("seed", 42)),  # 再現性向上
    )

    # モデルの準備（HuBERT重複ロード回避 - メモリ効率最適化）
    # 最初に一度だけHubertExtractorを作成
    shared_extractor = (
        HubertExtractor(
            model_name=cfg.model.hubert_model_name,
            layer=cfg.model.hubert_layer,
        )
        .float()
        .eval()
    )

    # 共有エクストラクターの勾配を無効化
    for param in shared_extractor.parameters():
        param.requires_grad = False

    # 作成したインスタンスを注入してFeatureCleanerを初期化（メモリ節約）
    model = FeatureCleaner(cfg.model, hubert_extractor=shared_extractor).float()

    # target_modelは共有エクストラクターを直接使用
    target_model = shared_extractor

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

    # 進捗追跡とチェックポイント管理の統一
    progress_bar = tqdm(range(cfg.steps), disable=not accelerator.is_main_process, desc="Training")
    completed_steps = 0

    # ★ 進捗バーとステップ数をAcceleratorの管理下に置く
    checkpoint_state = {"completed_steps": completed_steps}
    accelerator.register_for_checkpointing(progress_bar, checkpoint_state)

    # チェックポイントからの再開
    resume_from_checkpoint = cfg.checkpoint.get("resume_from")
    if resume_from_checkpoint:
        print_main(accelerator, f"Resuming from checkpoint: {resume_from_checkpoint}")
        accelerator.load_state(resume_from_checkpoint)
        completed_steps = checkpoint_state["completed_steps"]
        print_main(accelerator, f"Resuming from step {completed_steps}")
    elif pathlib.Path(cfg.save_dir).exists() and any(pathlib.Path(cfg.save_dir).iterdir()):
        # 自動検出による再開
        print_main(accelerator, f"Auto-resuming from {cfg.save_dir}")
        try:
            accelerator.load_state(cfg.save_dir)
            completed_steps = checkpoint_state["completed_steps"]
            print_main(accelerator, f"Auto-resumed from step {completed_steps}")
        except Exception as e:
            print_main(accelerator, f"Failed to auto-resume: {e}")
            print_main(accelerator, "Starting training from scratch")
            completed_steps = 0

    # データローダーのイテレーター作成
    train_iter = iter(train_dl)

    print_main(accelerator, "Starting Adapter training with Accelerate")
    print_main(accelerator, f"Mixed precision: {accelerator.mixed_precision}")
    print_main(accelerator, f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")

    # 学習ループ
    for step in range(completed_steps, cfg.steps):
        # 検証の実行
        if step > 0 and step % cfg.validation_interval == 0:
            val_losses = validate(model, target_model, val_dl, loss_fns, accelerator, cfg)
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

                # 損失の計算（サンプル長正規化改善）
                mae_loss = mae_loss_fn(pred, target)
                mse_loss = mse_loss_fn(pred, target)

                # Spectral Convergence Loss: サンプル長で正規化
                pred_power = pred.pow(2).mean(dim=(1, 2))  # (B,) - バッチ毎の平均パワー
                target_power = target.pow(2).mean(dim=(1, 2))  # (B,) - バッチ毎の平均パワー
                diff_power = (pred - target).pow(2).mean(dim=(1, 2))  # (B,) - バッチ毎の差分パワー
                sc_loss = (diff_power / (target_power + 1e-9)).mean()  # バッチ全体の平均

                loss = mae_loss + mse_loss + sc_loss

            # 逆伝播と最適化
            accelerator.backward(loss)

            # 勾配クリッピング
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=cfg.optim.max_grad_norm)
                # ★ 進捗バーとステップ数の更新
                progress_bar.update(1)
                checkpoint_state["completed_steps"] = step + 1

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

        # チェックポイントの保存（統一管理）
        if step > 0 and step % cfg.checkpoint.save_interval == 0:
            output_dir = pathlib.Path(cfg.save_dir) / f"checkpoint_{step}"
            print_main(accelerator, f"Saving unified checkpoint at step {step}")
            accelerator.save_state(output_dir, safe_serialization=False)

    # 学習終了時の処理
    print_main(accelerator, "Training completed")
    if accelerator.is_main_process:
        final_output_dir = pathlib.Path(cfg.save_dir) / "final"
        accelerator.save_state(final_output_dir, safe_serialization=False)
        progress_bar.close()
        accelerator.end_training()
