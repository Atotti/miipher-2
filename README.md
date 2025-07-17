# open-miipher-2

HuBERT + Parallel Adapter + Lightning SSL-Vocoder で [Miipher-2](https://arxiv.org/abs/2505.04457) を再現するリポジトリです。

## Prerequisites

```bash
uv sync
```

## Directory Structure

```
configs/           hydra yaml (preprocess, adapter, infer)
src/miipher_2/     python modules
cmd/               entry-point CLI wrappers
exp/               checkpoints 出力先
```

## データの前処理

### 擬似劣化データセットを生成

```bash
uv run cmds/preprocess.py --config-name preprocess
```
JVSコーパス形式ダウンロードした構造から直接処理可能。出力はwebdataset形式で保存される。

## モデル学習

### Parallel Adapter

```bash
uv run cmds/train_adapter.py --config-name adapter_l2
```

### Lightning SSL-Vocoder

SSL-vocoderはssl-vocodersリポジトリで学習します
学習済みモデルは推論設定ファイルで指定します。

### 学習の再開

特定のチェックポイントから再開
```bash
# Adapter学習の再開
uv run cmds/train_adapter.py checkpoint.resume_from="exp/adapter_l2/checkpoint_80k.pt" --config-name adapter_l2
```
※明示的に指定しない限り、checkpoint作成時のconfigが継承される

詳細な使用方法は [Checkpoint機能](docs/checkpoint_guide.md) を参照。


## 推論

### 設定ファイルの準備

### バッチ推論
```bash
uv run cmd/inference_dir.py --config-name infer_dir
```

## 評価

評価用劣化音声を生成
```bash
uv run cmd/degrade.py --clean_dir /home/ayu/GitHub/miipher-plaoground/samples --noise_dir /home/audio/TAU2023/dataset/TAU-urban-acoustic-scenes-2022-mobile-development/audio/ --out_dir /home/ayu/GitHub/miipher-plaoground/degrade_samples
```

劣化音声を復元
```bash
uv run cmd/inference_dir.py --config-name infer_dir
```

復元評価
```bash
uv run cmds/evaluate_simple.py --clean_dir /home/ayu/GitHub/miipher-plaoground/samples --restored_dir /home/ayu/GitHub/miipher-plaoground/hubert_large_l2/degrade_samples --outfile results/hubert_large_l2/degrade_samples.csv &&\
uv run cmds/evaluate_simple.py --clean_dir /home/ayu/GitHub/miipher-plaoground/samples --restored_dir /home/ayu/GitHub/miipher-plaoground/hubert_large_l2/samples_8khz_16khz --outfile results/hubert_large_l2/samples_8khz_16khz.csv &&\
\
uv run cmds/evaluate_simple.py --clean_dir /home/ayu/GitHub/miipher-plaoground/samples --restored_dir /home/ayu/GitHub/miipher-plaoground/mhubert_l6/degrade_samples --outfile results/mhubert_l6/degrade_samples.csv  &&\
uv run cmds/evaluate_simple.py --clean_dir /home/ayu/GitHub/miipher-plaoground/samples --restored_dir /home/ayu/GitHub/miipher-plaoground/mhubert_l6/samples_8khz_16khz --outfile results/mhubert_l6/samples_8khz_16khz.csv  &&\
\
uv run cmds/evaluate_simple.py --clean_dir /home/ayu/GitHub/miipher-plaoground/samples --restored_dir /home/ayu/GitHub/miipher-plaoground/wav2vec2_base_l2/degrade_samples --outfile results/wav2vec2_base_l2/degrade_samples.csv  &&\
uv run cmds/evaluate_simple.py --clean_dir /home/ayu/GitHub/miipher-plaoground/samples --restored_dir /home/ayu/GitHub/miipher-plaoground/wav2vec2_base_l2/samples_8khz_16khz --outfile results/wav2vec2_base_l2/samples_8khz_16khz.csv &&\
\
uv run cmds/evaluate_simple.py --clean_dir /home/ayu/GitHub/miipher-plaoground/samples --restored_dir /home/ayu/GitHub/miipher-plaoground/wavlm_base_l2/degrade_samples --outfile results/wavlm_base_l2/degrade_samples.csv  &&\
uv run cmds/evaluate_simple.py --clean_dir /home/ayu/GitHub/miipher-plaoground/samples --restored_dir /home/ayu/GitHub/miipher-plaoground/wavlm_base_l2/samples_8khz_16khz --outfile results/wavlm_base_l2/samples_8khz_16khz.csv

uv run cmds/evaluate_simple.py --clean_dir /home/ayu/GitHub/miipher-plaoground/samples --restored_dir /home/ayu/GitHub/miipher-plaoground/samples_8khz_16khz --outfile results/samples_8khz_16khz.csv &&\
uv run cmds/evaluate_simple.py --clean_dir /home/ayu/GitHub/miipher-plaoground/samples --restored_dir /home/ayu/GitHub/miipher-plaoground/degrade_samples --outfile results/degrade_samples.csv
```
