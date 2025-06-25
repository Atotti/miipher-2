import pathlib

import hydra
import torch
import torchaudio
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from miipher_2.extractors.hubert import HubertExtractor

# このスクリプトは、preprocess.yamlで定義されたデータセットを読み込み、
# 各音声に対応するHuBERT特徴量を計算して保存します。

@hydra.main(version_base=None, config_path="../configs", config_name="cache_features")
def main(cfg: DictConfig) -> None:
    print("--- Initializing Feature Caching ---")

    # データセットのインスタンス化 (e.g., JVSCorpus)
    dataset_cfg = cfg.preprocess.preprocess_dataset.datasets[0]
    dataset = hydra.utils.instantiate(dataset_cfg)
    print(f"Loaded dataset: {type(dataset).__name__} with {len(dataset)} files.")

    # HubertExtractorの準備
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hubert_extractor = HubertExtractor(
        model_name=cfg.cache.hubert_model_name,
        layer=cfg.cache.hubert_layer,
    ).to(device).eval()
    print(f"HubertExtractor initialized on {device}.")

    # 出力ディレクトリの作成
    output_dir = pathlib.Path(cfg.cache.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache will be saved to: {output_dir}")

    # データローダーの作成
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.cache.batch_size,
        num_workers=cfg.cache.num_workers,
        collate_fn=lambda batch: batch,  # 個別のサンプルをそのままリストで返す
    )

    # 特徴量抽出と保存
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Caching HuBERT features"):
            wav_paths = [item["wav_path"] for item in batch]
            basenames = [item["basename"] for item in batch]

            # 音声ファイルをロードし、16kHzにリサンプリング
            wavs_16k = []
            for wav_path in wav_paths:
                wav, sr = torchaudio.load(wav_path)
                wav = wav.mean(0, keepdim=True)  # モノラル化
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
                wavs_16k.append(wav.squeeze(0))  # (T,)の形状にする

            # バッチ処理のためにパディング
            padded_wavs = torch.nn.utils.rnn.pad_sequence(wavs_16k, batch_first=True).to(device)

            # HuBERT特徴量を抽出
            features = hubert_extractor(padded_wavs)  # (B, C, T_feat)

            # 各サンプルの特徴量を個別のファイルとして保存
            for i, feat in enumerate(features):
                # パディング前の長さを計算
                original_len = wavs_16k[i].size(0)
                # HuBERTの入力から出力への変換比率（通常320:1）
                feat_len = 320  # HuBERTの標準的なhop_length
                unpadded_feat_len = (original_len + feat_len - 1) // feat_len

                # パディング部分を削除してCPUに移動
                unpadded_feat = feat[:, :unpadded_feat_len].cpu()

                # 保存
                output_path = output_dir / f"{basenames[i]}.pt"
                torch.save(unpadded_feat, output_path)

    print("--- Feature Caching Completed ---")


if __name__ == "__main__":
    main()
