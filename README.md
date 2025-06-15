# open‑miipher‑2

mHuBERT‑147 (9 th) + Parallel Adapter + HiFi‑GAN で
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

### 音声を用意

* **クリーン音声** (16 kHz mono, WAV) を `/home/ayu/datasets/` へ配置
  例：FLEURS‑R, LibriTTS‑R, JVS など

### 擬似劣化データセットを生成

```bash
uv run cmd/preprocess.py --config-name preprocess
```

上記は `configs/preprocess.yaml` がベース。
生成物は **webdataset** 形式の tar 連番です。


## 2️⃣ モデル学習

###  Parallel Adapter (feature cleaner)

```bash
uv run cmd/train_hubert.py --config-name adapter
```

### HiFi‑GAN fine‑tune

```bash
uv run cmd/train_vocoder.py --config-name hifigan_finetune

```


## 3️⃣ 推論

```bash
uv run cmd/inference.py \
    adapter=exp/adapter/adapter_final.pt \
    vocoder=exp/hifigan_ft/hifigan_ft_final.pt \
    in=noisy.wav out=restored.wav
```



## 4️⃣ 自動評価 (DNSMOS / SQuId / WER / SPK)

```bash
uv run cmd/evaluate.py \
    clean_dir=data/clean_test \
    rest_dir=restored_test
```
