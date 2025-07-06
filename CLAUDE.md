# CLAUDE.md

このファイルは、このリポジトリでコードを作業する際にClaude Code (claude.ai/code) にガイダンスを提供します。

## 開発コマンド

### 環境セットアップ
```bash
uv sync
```

### リンター・フォーマッター
```bash
# コードの品質チェック
uv run ruff check .

# コードフォーマット
uv run ruff format .

# 型チェック
uv run mypy .
```

### データ前処理
```bash
# JVSコーパスの前処理
uv run cmd/preprocess.py --config-name preprocess_jvs

# LibriTTSの前処理
uv run cmd/preprocess.py --config-name preprocess_libritts_r

# FLEURSの前処理
uv run cmd/preprocess.py --config-name preprocess_fleurs_r
```

### モデル学習
```bash
# Parallel Adapter学習
uv run cmd/train_adapter.py --config-name adapter_layer_6_mhubert_147

# 学習再開（特定のチェックポイントから）
uv run cmd/train_adapter.py checkpoint.resume_from="exp/adapter_layer_6_mhubert_147/checkpoint_199k.pt" --config-name adapter_layer_6_mhubert_147

# Lightning SSL-Vocoder事前学習
uv run cmd/pre_train_vocoder.py --config-name hifigan_pretrain_layer_6_mhubert_147
```

### 推論・評価
```bash
# バッチ推論
uv run cmd/inference_dir.py --config-name infer_dir

# 評価用劣化音声生成
uv run cmd/degrade.py --clean_dir <clean_dir> --noise_dir <noise_dir> --out_dir <out_dir>

# 評価実行
uv run cmd/evaluate.py --clean_dir <clean_dir> --degraded_dir <degraded_dir> --restored_dir <restored_dir> --outfile <output_csv>
```

## アーキテクチャ概要

### プロジェクト構造
- `src/miipher_2/`: メインのPythonモジュール
- `cmd/`: CLI エントリーポイント
- `configs/`: Hydra設定ファイル (YAML)
- `exp/`: 学習チェックポイント出力先

### 主要コンポーネント

#### 1. Parallel Adapter (`src/miipher_2/adapters/parallel_adapter.py`)
- HuBERTの特定層に挿入される軽量なフィードフォワードネットワーク
- LayerNorm + Linear + GELU + Linear の構成
- Xavier初期化を0.01スケールで適用

#### 2. HuBERT Feature Extractor (`src/miipher_2/extractors/hubert.py`)
- 事前学習済みmHuBERT-147モデルを使用
- 指定した層(デフォルト6層)の特徴量を抽出
- Parallel Adapterが特定層に挿入される

#### 3. Lightning SSL-Vocoder (`src/miipher_2/lightning_vocoders/`)
- HiFi-GANベースの音声合成モデル
- PyTorch Lightningでの実装
- SSL特徴量からメルスペクトログラムを生成

#### 4. WebDataset Loader (`src/miipher_2/data/webdataset_loader.py`)
- 大規模データセットの効率的な読み込み
- WebDataset形式での並列処理
- 動的なバッチ生成とシャッフル

#### 5. 学習・推論パイプライン
- `src/miipher_2/train/adapter.py`: Adapter学習ロジック
- `src/miipher_2/utils/infer.py`: 推論実行ロジック
- Wandb統合による学習監視

### 設定管理
- Hydra設定を使用してハイパーパラメータを管理
- 主要設定: `configs/adapter_layer_6_mhubert_147.yaml`
- チェックポイント機能による学習再開サポート

### データフロー
1. 音声データをWebDataset形式で前処理
2. HuBERTで特徴抽出（Parallel Adapter適用）
3. 抽出した特徴量でFeature Cleanerを学習
4. Lightning SSL-Vocoderで音声合成
5. 複数の評価指標で性能測定