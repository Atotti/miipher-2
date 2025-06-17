# open‑miipher‑2

HuBERT + Parallel Adapter + HiFi‑GAN で [Miipher‑2](https://arxiv.org/abs/2505.04457) を再現するリポジトリです。

## Prerequisites

```bash
uv sync
```

## Directory Structure

```
configs/           hydra yaml (preprocess, adapter, hifigan, infer)
src/miipher_2/     python modules
cmd/               entry‑point CLI wrappers
exp/               checkpoints 出力先
```

## データの前処理

### 擬似劣化データセットを生成

```bash
uv run cmd/preprocess.py --config-name preprocess
```
JVSコーパス形式ダウンロードした構造から直接処理可能。出力はwebdataset形式で保存される。

## モデル学習

### Parallel Adapter

```bash
uv run cmd/train_adapter.py --config-name adapter
```

### HiFi‑GAN fine‑tune

```bash
uv run cmd/train_vocoder.py --config-name hifigan_finetune
```

### 学習の再開

特定のチェックポイントから再開
```bash
# Adapter学習の再開
uv run cmd/train_adapter.py checkpoint.resume_from="exp/adapter/checkpoint_5k.pt"

# HiFi-GAN学習の再開
uv run cmd/train_vocoder.py checkpoint.resume_from="exp/hifigan_ft/checkpoint_10k.pt"
```
※明示的に指定しない限り、checkpoint作成時のconfigが継承される

詳細な使用方法は [Checkpoint機能](docs/checkpoint_guide.md) を参照。


## 推論

```bash
uv run cmd/inference.py --config-name infer
```


## 自動評価 (DNSMOS / SQuId / WER / SPK)

```bash
uv run cmd/evaluate.py --config-name evaluate
```
