import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

import wandb


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    scheduler_state: dict[str, Any] | None = None,
    epoch: int | None = None,
    additional_states: dict[str, Any] | None = None,
    cfg: DictConfig | None = None,
    keep_last_n: int = 5,
) -> str:
    """
    チェックポイントを保存する

    Args:
        checkpoint_dir: チェックポイント保存ディレクトリ
        step: 現在のステップ数
        model_state: モデルの状態辞書
        optimizer_state: オプティマイザの状態辞書
        scheduler_state: スケジューラの状態辞書
        epoch: エポック数
        additional_states: 追加の状態辞書
        cfg: 設定
        keep_last_n: 保持するチェックポイント数

    Returns:
        保存されたチェックポイントのパス
    """
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    # チェックポイントファイル名
    checkpoint_name = f"checkpoint_{step // 1000}k.pt"
    checkpoint_path = checkpoint_dir_path / checkpoint_name

    # 保存するデータ
    checkpoint_data = {
        "step": step,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
    }

    if scheduler_state is not None:
        checkpoint_data["scheduler_state_dict"] = scheduler_state

    if epoch is not None:
        checkpoint_data["epoch"] = epoch

    if additional_states is not None:
        checkpoint_data.update(additional_states)

    # Wandb情報を保存
    if wandb.run is not None:
        checkpoint_data["wandb_run_id"] = wandb.run.id
        checkpoint_data["wandb_run_name"] = wandb.run.name
        checkpoint_data["wandb_project"] = wandb.run.project

    # 設定を保存
    if cfg is not None:
        checkpoint_data["config"] = dict(cfg)

    # 乱数状態を保存
    checkpoint_data["random_states"] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint_data["random_states"]["torch_cuda"] = torch.cuda.get_rng_state_all()

    # チェックポイント保存
    torch.save(checkpoint_data, checkpoint_path)
    print(f"[INFO] Checkpoint saved: {checkpoint_path}")

    # 古いチェックポイントを削除
    cleanup_old_checkpoints(checkpoint_dir_path, keep_last_n)

    return str(checkpoint_path)


def load_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """
    チェックポイントを読み込む

    Args:
        checkpoint_path: チェックポイントファイルのパス

    Returns:
        チェックポイントデータ
    """
    if not Path(checkpoint_path).exists():
        msg = f"Checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print(f"[INFO] Checkpoint loaded: {checkpoint_path}")
        return checkpoint  # noqa: TRY300
    except Exception as e:
        msg = f"Failed to load checkpoint {checkpoint_path}: {e}"
        raise RuntimeError(msg) from e


def validate_config_compatibility(checkpoint_config: dict[str, Any], current_config: dict[str, Any]) -> list[str]:
    """
    設定の互換性をチェックし、重要なパラメータの変更を検出

    Args:
        checkpoint_config: チェックポイントに保存された設定
        current_config: 現在の設定

    Returns:
        変更された重要なパラメータのリスト
    """
    critical_params = [
        "model",
        "optim.lr",
        "batch_size",
        "dataset.num_examples",
        "epochs",
        "steps",  # HiFi-GAN用
    ]

    warnings = []

    def get_nested_value(config: dict[str, Any], key: str) -> Any | None:  # noqa: ANN401
        """ネストされた設定値を取得"""
        keys = key.split(".")
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        return value

    for param in critical_params:
        checkpoint_val = get_nested_value(checkpoint_config, param)
        current_val = get_nested_value(current_config, param)

        if checkpoint_val is not None and current_val is not None and checkpoint_val != current_val:
            warnings.append(f"{param}: {checkpoint_val} -> {current_val}")

    return warnings


def setup_wandb_resume(cfg: DictConfig, checkpoint: dict[str, Any] | None = None) -> None:
    """
    Wandbの初期化と再開設定

    Args:
        cfg: 設定
        checkpoint: チェックポイントデータ
    """
    if not cfg.wandb.enabled:
        return

    if checkpoint is not None and "wandb_run_id" in checkpoint:
        # チェックポイントから自動的にWandb IDを継承
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            id=checkpoint["wandb_run_id"],
            resume="must",
            config=dict(cfg),  # 現在の設定を使用
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes,
        )
        print(f"[INFO] Resumed wandb run: {checkpoint['wandb_run_id']}")

        # 設定の互換性をチェック
        if "config" in checkpoint:
            config_warnings = validate_config_compatibility(checkpoint["config"], dict(cfg))
            if config_warnings:
                print("[WARNING] Configuration changes detected:")
                for warning in config_warnings:
                    print(f"  - {warning}")
                print("These changes may affect training consistency.")
    else:
        # 新規学習
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            config=dict(cfg),
            notes=cfg.wandb.notes,
        )
        print(f"[INFO] Started new wandb run: {wandb.run.id}")


def restore_random_states(checkpoint: dict[str, Any]) -> None:
    """
    乱数状態を復元する

    Args:
        checkpoint: チェックポイントデータ
    """
    if "random_states" not in checkpoint:
        print("[WARNING] No random states found in checkpoint")
        return

    random_states = checkpoint["random_states"]

    try:
        random.setstate(random_states["python"])
        np.random.set_state(random_states["numpy"])  # noqa: NPY002
        torch.set_rng_state(random_states["torch"])

        if torch.cuda.is_available() and "torch_cuda" in random_states:
            torch.cuda.set_rng_state_all(random_states["torch_cuda"])

        print("[INFO] Random states restored")
    except Exception as e:  # noqa: BLE001
        print(f"[WARNING] Failed to restore random states: {e}")


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int) -> None:
    """
    古いチェックポイントを削除する

    Args:
        checkpoint_dir: チェックポイントディレクトリ
        keep_last_n: 保持するチェックポイント数
    """
    if keep_last_n <= 0:
        return

    # チェックポイントファイルを取得
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pt"))

    if len(checkpoint_files) <= keep_last_n:
        return

    checkpoint_files.sort(key=os.path.getmtime, reverse=True)

    # 古いファイルを削除
    files_to_delete = checkpoint_files[keep_last_n:]
    for file_path in files_to_delete:
        try:
            Path(file_path).unlink()
            print(f"[INFO] Removed old checkpoint: {file_path}")
        except Exception as e:  # noqa: BLE001
            print(f"[WARNING] Failed to remove checkpoint {file_path}: {e}")


def find_latest_checkpoint(checkpoint_dir: str) -> Path | None:
    """
    最新のチェックポイントを見つける

    Args:
        checkpoint_dir: チェックポイントディレクトリ

    Returns:
        最新のチェックポイントパス
    """
    checkpoint_files = list(Path(checkpoint_dir).glob("checkpoint_*.pt"))
    if not checkpoint_files:
        return None

    # 最新のファイルを返す
    return max(checkpoint_files, key=os.path.getmtime)


def get_resume_checkpoint_path(cfg: DictConfig) -> Path | None:
    """
    再開用チェックポイントパスを取得する

    Args:
        cfg: 設定

    Returns:
        チェックポイントパス
    """
    if hasattr(cfg, "checkpoint") and cfg.checkpoint.resume_from:
        checkpoint_path = cfg.checkpoint.resume_from
        if Path(checkpoint_path).exists():
            return checkpoint_path
        print(f"[WARNING] Specified checkpoint not found: {checkpoint_path}")

    # 最新のチェックポイントを自動検索
    latest_checkpoint = find_latest_checkpoint(cfg.save_dir)
    if latest_checkpoint:
        print(f"[INFO] Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint

    return None
