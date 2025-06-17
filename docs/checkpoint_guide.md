# Checkpoint機能

Miipher-2の学習におけるチェックポイント機能の使用方法について説明します。

## 概要

チェックポイント機能により、以下が可能になります：

- **自動保存**: 1,000ステップごとに学習状態を自動保存
- **学習再開**: 中断された学習を正確な状態から再開
- **Wandb連携**: 学習履歴の連続性を保持
- **自動クリーンアップ**: 古いチェックポイントの自動削除

## 設定

### Adapter学習の設定 (`configs/adapter.yaml`)

```yaml
# Checkpoint configuration
checkpoint:
  save_interval: 1000      # 1kステップごとにチェックポイント保存
  resume_from: null        # 再開用チェックポイントパス
  keep_last_n: 5          # 最新N個のチェックポイントを保持
  save_wandb_metadata: true # wandb情報も保存

# Wandb logging configuration
wandb:
  enabled: true
  project: "miipher-2-adapter"
  entity: null
  name: null
  tags: ["adapter", "training"]
  notes: "Parallel Adapter training for Miipher-2"
  log_model: false
```

### HiFi-GAN学習の設定 (`configs/hifigan_finetune.yaml`)

```yaml
# Checkpoint configuration
checkpoint:
  save_interval: 1000      # 1kステップごとにチェックポイント保存
  resume_from: null        # 再開用チェックポイントパス
  keep_last_n: 5          # 最新N個のチェックポイントを保持
  save_wandb_metadata: true # wandb情報も保存

# Wandb logging configuration
wandb:
  enabled: true
  project: "miipher-2-hifigan"
  entity: null
  name: null
  tags: ["hifigan", "vocoder", "finetune"]
  notes: "HiFi-GAN fine-tuning for Miipher-2"
  log_model: true
  log_audio: true
```

## 使用方法

### 1. 通常の学習開始

```bash
# Adapter学習
uv run cmd/train_adapter.py

# HiFi-GAN学習
uv run cmd/train_vocoder.py
```

### 2. 特定のチェックポイントから再開

```bash
# Adapter学習
uv run cmd/train_adapter.py checkpoint.resume_from="exp/adapter/checkpoint_5k.pt"

# HiFi-GAN学習
uv run cmd/train_vocoder.py checkpoint.resume_from="exp/hifigan_ft/checkpoint_10k.pt"
```

### 3. 自動再開スクリプトの使用

最新のチェックポイントを自動で見つけて再開：

```bash
# Adapter学習の自動再開
bash scripts/auto_resume_adapter.sh

# HiFi-GAN学習の自動再開
bash scripts/auto_resume_hifigan.sh
```



## チェックポイントファイルの構造

### Adapter学習のチェックポイント

```python
{
    'step': 5000,                           # 現在のステップ数
    'epoch': 2,                             # 現在のエポック数
    'model_state_dict': {...},              # モデルの状態
    'optimizer_state_dict': {...},          # オプティマイザの状態
    'scheduler_state_dict': {...},          # スケジューラの状態
    'wandb_run_id': 'abc123',              # Wandb Run ID
    'wandb_run_name': 'adapter_run_1',     # Wandb Run名
    'wandb_project': 'miipher-2-adapter',  # Wandbプロジェクト名
    'config': {...},                        # 学習設定
    'random_states': {                      # 乱数状態（再現性のため）
        'python': ...,
        'numpy': ...,
        'torch': ...,
        'torch_cuda': ...
    }
}
```

### HiFi-GAN学習のチェックポイント

```python
{
    'step': 10000,                          # 現在のステップ数
    'model_state_dict': {...},              # Generator状態
    'optimizer_state_dict': {...},          # Generator Optimizer状態
    'mpd_state_dict': {...},               # Multi-Period Discriminator状態
    'msd_state_dict': {...},               # Multi-Scale Discriminator状態
    'opt_d_state_dict': {...},             # Discriminator Optimizer状態
    'scaler_state_dict': {...},            # AMP Scaler状態
    'wandb_run_id': 'def456',              # Wandb Run ID
    'wandb_run_name': 'hifigan_run_1',     # Wandb Run名
    'config': {...},                        # 学習設定
    'random_states': {...}                  # 乱数状態
}
```

## Wandb連携

### 自動ID継承（デフォルト）

チェックポイントを指定して学習を再開すると：

- **自動的にWandb IDが継承**される
- 同じWandb RunIDで学習を継続
- 学習曲線が途切れない
- メトリクスの連続性が保たれる

### 設定の互換性チェック

学習再開時に重要なパラメータの変更を自動検出：

```
[WARNING] Configuration changes detected:
  - optim.lr: 0.0002 -> 0.0001
  - batch_size: 16 -> 32
These changes may affect training consistency.
```

チェックされるパラメータ：
- `model`: モデル構造
- `optim.lr`: 学習率
- `batch_size`: バッチサイズ
- `dataset.num_examples`: データセット例数
- `epochs`: エポック数
- `steps`: ステップ数（HiFi-GAN）

## ファイル管理

### 自動クリーンアップ

`keep_last_n: 5`の設定により：

- 最新5個のチェックポイントのみ保持
- 古いチェックポイントは自動削除
- ディスク容量の節約

### ファイル命名規則

```
exp/adapter/checkpoint_1k.pt    # 1,000ステップ
exp/adapter/checkpoint_2k.pt    # 2,000ステップ
exp/adapter/checkpoint_3k.pt    # 3,000ステップ
...
```

## 高度な使用方法

### カスタムチェックポイント間隔

```yaml
checkpoint:
  save_interval: 500  # 500ステップごとに保存
```

### 特定のチェックポイントのみ保持

```yaml
checkpoint:
  keep_last_n: 10  # 最新10個を保持
```

### Wandbメタデータの無効化

```yaml
checkpoint:
  save_wandb_metadata: false
```
