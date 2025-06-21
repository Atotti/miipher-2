# チェックポイント再開ガイド

このドキュメントでは、Miipher-2のトレーニングでチェックポイントから再開する方法について説明します。

## 基本的な動作

**デフォルトの動作**: 明示的にチェックポイントを指定しない限り、常に新規学習を開始します。

## チェックポイントから再開する方法

### 1. 設定ファイルで指定

設定ファイル（`.yaml`）の`checkpoint.resume_from`にチェックポイントのパスを指定します：

```yaml
checkpoint:
  save_interval: 1000
  resume_from: "exp/adapter_layer_4_mhubert_147/checkpoint_10k.pt"  # ここにパスを指定
  keep_last_n: 500
  save_wandb_metadata: true
```

### 2. コマンドラインで指定

Hydraの機能を使用してコマンドラインから指定することも可能です：

```bash
# Adapterの学習を再開
uv run cmd/train_adapter.py checkpoint.resume_from="exp/adapter_layer_4_mhubert_147/checkpoint_10k.pt"

# HiFi-GANの事前学習を再開
uv run cmd/pre_train_vocoder.py checkpoint.resume_from="exp/hifigan_pretrain_layer_6_mhubert_147/checkpoint_5k.pt"

# HiFi-GANのファインチューニングを再開
uv run cmd/finetune_vocoder.py checkpoint.resume_from="exp/hifigan_ft_layer_6_mhubert_147/checkpoint_15k.pt"
```

## ログメッセージ

### 新規学習開始時
```
[INFO] Starting fresh training (no checkpoint specified)
```

### チェックポイントから再開時
```
[INFO] Resuming from explicitly specified checkpoint: /path/to/checkpoint.pt
[INFO] Resuming from step 10000
[INFO] Restored model, optimizer, and scheduler states
```

## 注意事項

1. **明示的な指定が必要**: `checkpoint.resume_from`を明示的に指定しない限り、チェックポイントディレクトリに既存のチェックポイントがあっても自動的には再開されません。

2. **チェックポイントの互換性**: 再開するチェックポイントは、現在の設定と互換性がある必要があります。モデルアーキテクチャやハイパーパラメータが大きく異なる場合、エラーが発生する可能性があります。

3. **Wandb ID の継承**: チェックポイントから再開する場合、Wandb run IDも自動的に継承され、同じ実験として継続されます。

4. **乱数状態の復元**: チェックポイントから再開する際、Python、NumPy、PyTorchの乱数状態も復元されるため、再現可能な学習が保証されます。

## トラブルシューティング

### チェックポイントが見つからない場合
```
FileNotFoundError: Checkpoint specified in `resume_from` not found: /path/to/checkpoint.pt
```
→ パスが正しいか確認してください。

### 設定の互換性警告
```
[WARNING] Configuration changes detected:
  - model.hubert_layer: 4 -> 6
  - batch_size: 16 -> 32
These changes may affect training consistency.
```
→ 重要なパラメータが変更されている場合に表示されます。学習の一貫性に影響する可能性があります。

## 例

### Adapterの学習を10,000ステップから再開
```bash
uv run cmd/train_adapter.py \
  --config-name=adapter_layer_4_mhubert_147 \
  checkpoint.resume_from="exp/adapter_layer_4_mhubert_147/checkpoint_10k.pt"
```

### HiFi-GANの事前学習を5,000ステップから再開
```bash
uv run cmd/pre_train_vocoder.py \
  --config-name=hifigan_pretrain_layer_6_mhubert_147 \
  checkpoint.resume_from="exp/hifigan_pretrain_layer_6_mhubert_147/checkpoint_5k.pt"
