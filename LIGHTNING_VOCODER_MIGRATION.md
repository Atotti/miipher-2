# Lightning SSL-Vocoder Migration Guide

## 変更点

Miipher-2 の推論機能を HiFiGAN から Lightning SSL-Vocoder に移行しました。

### 主な変更

1. **HiFiGAN + PreNet** → **Lightning SSL-Vocoder** (Conformer + HiFiGAN統合)
2. 設定ファイルの簡素化（PreNet設定が不要）
3. チェックポイント形式の変更（`.pt` → `.ckpt`）

## 必要な準備

### 1. SSL-Vocoder モデルの準備

`/home/ayu/GitHub/ssl-vocoders` で学習したモデルを使用してください。

```bash
# ssl-vocodersでの学習例
cd /home/ayu/GitHub/ssl-vocoders
python train.py --config config/wavlm_large.yaml
```

### 2. 設定ファイルの更新

#### Before (旧HiFiGAN設定):
```yaml
vocoder_ckpt: "exp/hifigan_pretrain_layer_4/checkpoint_96k.pt"
prenet:
  in_dim: 768
  n_layers: 4
  mel_dim: 80
  src_fps: 50.0
  tgt_hop: 256
  sr: 22050
```

#### After (Lightning SSL-Vocoder設定):
```yaml
vocoder_ckpt: "/path/to/ssl-vocoder-checkpoint.ckpt"
# PreNet設定は不要（Lightning SSL-Vocoderが内部で処理）
```

## 使用方法

### 単一ファイルの推論

```bash
# 設定ファイルのvocoder_ckptを更新
vim configs/infer.yaml

# 推論実行
uv run python -m miipher_2.infer --config configs/infer.yaml
```

### バッチ推論

```bash
# 設定ファイルのvocoder_ckptを更新
vim configs/infer_dir.yaml

# バッチ推論実行
uv run python -m miipher_2.infer_dir --config configs/infer_dir.yaml
```

## 推奨モデル

SSL-Vocodersで利用可能な事前学習済みモデル：

| モデル | SSL特徴量 | 品質 | 用途 |
|--------|-----------|------|------|
| `wavlm-large` | WavLM Large | 最高 | 高品質推論 |
| `hubert-base` | HuBERT Base | 良好 | 軽量推論 |
| `wav2vec2-base` | Wav2Vec2.0 Base | 良好 | 汎用 |

## 注意事項

1. **チェックポイント形式**: `.pt` → `.ckpt` に変更
2. **PreNet設定削除**: Lightning SSL-Vocoderが内部で処理
3. **依存関係**: `lightning` パッケージが必要
4. **SSL特徴量の一致**: FeatureCleanerとSSL-Vocoderで同じSSL特徴量を使用

## トラブルシューティング

### ImportError: lightning_vocoders
```bash
# __init__.py と hifigan.py が正しくコピーされているか確認
ls src/miipher_2/lightning_vocoders/
```

### CheckpointNotFound
```bash
# パスが正しいか確認
ls -la /path/to/ssl-vocoder-checkpoint.ckpt
```

### SSL特徴量の不一致
```bash
# FeatureCleanerとSSL-Vocoderで同じHuBERTレイヤを使用しているか確認
```