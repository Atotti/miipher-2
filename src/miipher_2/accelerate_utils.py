import os
import random
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.tracking import WandBTracker
from omegaconf import DictConfig, OmegaConf


def worker_init_fn(worker_id: int, seed: int = 42) -> None:
    """
    DataLoaderワーカーの乱数シードを設定する関数

    Args:
        worker_id: ワーカーID
        seed: ベースシード
    """
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_accelerator(cfg: DictConfig) -> tuple[Accelerator, Any]:
    """
    Acceleratorとロガーを構築する共通関数

    Args:
        cfg: 設定ファイル（Hydra/OmegaConf）

    Returns:
        tuple[Accelerator, Logger]: AcceleratorとLogger
    """
    # Acceleratorの初期化
    accelerator = Accelerator(
        mixed_precision=cfg.training.get("mixed_precision", "no"),  # "fp16" / "bf16" / "no"
        gradient_accumulation_steps=cfg.training.get("gradient_accumulation_steps", 1),
        log_with="wandb" if cfg.wandb.get("enabled", False) else None,
        project_dir=cfg.save_dir,
        cpu=cfg.training.get("force_cpu", False),
    )

    # ログレベルの設定
    logger = get_logger(__name__, log_level="INFO" if accelerator.is_main_process else "WARNING")

    # WandBトラッカーの初期化（メインプロセスのみ）
    if accelerator.is_main_process and cfg.wandb.get("enabled", False):
        # OmegaConfをdictに変換
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        accelerator.init_trackers(
            project_name=cfg.wandb.project,
            config=config_dict,
            init_kwargs={
                "wandb": {
                    "name": cfg.wandb.get("name", None),
                    "tags": cfg.wandb.get("tags", []),
                    "notes": cfg.wandb.get("notes", ""),
                    "entity": cfg.wandb.get("entity", None),
                    "resume": "allow",  # チェックポイントからの再開を許可
                    "id": cfg.wandb.get("id", None),  # 固定IDで再開
                }
            },
        )
        logger.info("WandB tracker initialized")

    # デバイス情報をログ出力
    if accelerator.is_main_process:
        logger.info(f"Using device: {accelerator.device}")
        logger.info(f"Mixed precision: {accelerator.mixed_precision}")
        logger.info(f"Distributed type: {accelerator.distributed_type}")
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info(f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")

    return accelerator, logger


def setup_random_seeds(accelerator: Accelerator, seed: int = 42) -> None:
    """
    すべてのプロセスで同期された乱数シードを設定

    Args:
        accelerator: Acceleratorインスタンス
        seed: 乱数シード
    """
    accelerator.wait_for_everyone()
    accelerator.seed_everything(seed)


def safe_save_model(accelerator: Accelerator, model: Any, path: str) -> None:
    """
    Accelerate対応のモデル保存（メインプロセスのみ実行）

    Args:
        accelerator: Acceleratorインスタンス
        model: 保存するモデル
        path: 保存先パス
    """
    if accelerator.is_main_process:
        # unwrap_modelでDataParallel/DistributedDataParallelをunwrap
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), path)


def get_device_from_accelerator(accelerator: Accelerator) -> str:
    """
    Acceleratorからデバイス情報を取得

    Args:
        accelerator: Acceleratorインスタンス

    Returns:
        str: デバイス名
    """
    return str(accelerator.device)


def log_metrics(accelerator: Accelerator, metrics: dict[str, float], step: int) -> None:
    """
    メトリクスをロギング（メインプロセスのみ）

    Args:
        accelerator: Acceleratorインスタンス
        metrics: ログに記録するメトリクス
        step: ステップ数
    """
    if accelerator.is_main_process:
        accelerator.log(metrics, step=step)


def print_main(accelerator: Accelerator, message: str) -> None:
    """
    メインプロセスのみでprint

    Args:
        accelerator: Acceleratorインスタンス
        message: 出力メッセージ
    """
    if accelerator.is_main_process:
        print(message)
