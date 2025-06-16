# open‑miipher‑2

HuBERT + Parallel Adapter + HiFi‑GAN で
Miipher‑2 を再現するリポジトリです。

## 🍀 Prerequisites

```bash
uv sync
```

## 🏗️ Directory Structure

```
configs/           hydra yaml (preprocess, adapter, hifigan, infer)
src/miipher_2/     python modules
cmd/               entry‑point CLI wrappers
exp/               checkpoints 出力先
```

## 1️⃣ データの前処理

### 擬似劣化データセットを生成

```bash
uv run cmd/preprocess.py --config-name preprocess
```
JVSコーパス形式ダウンロードした構造から直接処理可能。出力はwebdataset形式で保存される。

## 2️⃣ モデル学習

###  Parallel Adapter

```bash
uv run cmd/train_adapter.py --config-name adapter
```

### HiFi‑GAN fine‑tune

```bash
uv run cmd/train_vocoder.py --config-name hifigan_finetune

```

## 3️⃣ 推論

```bash
uv run cmd/inference.py --config-name infer
```


## 4️⃣ 自動評価 (DNSMOS / SQuId / WER / SPK)

```bash
uv run cmd/evaluate.py --config-name evaluate
```

