import json
import pathlib

import torch
import torch.amp
from omegaconf import DictConfig
from tqdm import tqdm

from miipher_2.prenet.models import Generator
from miipher_2.prenet.prenet import Miipher2PreNet
from miipher_2.model.feature_cleaner import FeatureCleaner
from miipher_2.train.hifigan import AttrDict
from miipher_2.utils.audio import load, save


@torch.inference_mode()
def run_inference(cfg: DictConfig) -> None:
    """
    設定ファイルに基づいてMiipher-2の音声修復推論を実行する

    Args:
        cfg (DictConfig): Hydraによって読み込まれた設定オブジェクト
    """
    device = torch.device(cfg.device)

    # 1. FeatureCleaner ... (変更なし)
    print("Loading FeatureCleaner model...")
    cleaner = FeatureCleaner(cfg.model).to(device).eval()
    adapter_checkpoint = torch.load(cfg.adapter_ckpt, map_location=device, weights_only=False)
    cleaner.load_state_dict(adapter_checkpoint["model_state_dict"])
    print("FeatureCleaner model loaded.")

    # 2. Vocoder ... (変更なし)
    print("Loading Vocoder models (Prenet + Generator)...")
    cfg.prenet.in_dim = cleaner.extractor.hubert.config.hidden_size
    prenet = Miipher2PreNet(**cfg.prenet).to(device).eval()
    vocoder_ckpt_path = pathlib.Path(cfg.vocoder_ckpt)
    vocoder_config_path = vocoder_ckpt_path.parent / "config.json"
    if not vocoder_config_path.exists():
        msg = (
            f"Vocoder config file not found at: {vocoder_config_path}\n"
            "Please make sure 'config.json' is present in the same directory as the vocoder checkpoint."
        )
        raise FileNotFoundError(msg)
    with vocoder_config_path.open() as f:
        h = AttrDict(json.load(f))
    generator = Generator(h).to(device).eval()
    vocoder_checkpoint = torch.load(vocoder_ckpt_path, map_location=device, weights_only=False)
    prenet.load_state_dict(vocoder_checkpoint["prenet"])
    generator.load_state_dict(vocoder_checkpoint["generator"])
    print("Vocoder models loaded.")

    # 3. 音声ファイルを読み込み、推論実行
    print(f"Processing input file: {cfg.input_wav}")
    input_wav = load(cfg.input_wav).to(device)

    # 非推奨警告に対応
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
        cleaned_features = cleaner(input_wav)
        mel_spectrogram = prenet(cleaned_features)
        restored_wav = generator(mel_spectrogram)

    # 4. 修復された音声を保存
    output_path = pathlib.Path(cfg.output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 【修正点】 .squeeze(0) でバッチ次元を削除
    save(output_path, restored_wav.squeeze(0).cpu().to(torch.float32), sr=cfg.output_sampling_rate)

    print(f"Restored audio saved to: {output_path}")


@torch.inference_mode()
def run_inference_dir(cfg: DictConfig) -> None:
    """
    ディレクトリ内の全ての音声ファイルに対して一括で音声修復推論を実行する

    Args:
        cfg (DictConfig): Hydraによって読み込まれた設定オブジェクト
    """
    device = torch.device(cfg.device)

    # 1. モデルの読み込み (ループの外で一度だけ行います)
    print("Loading models...")
    cleaner = FeatureCleaner(cfg.model).to(device).eval()
    adapter_checkpoint = torch.load(cfg.adapter_ckpt, map_location=device, weights_only=False)
    cleaner.load_state_dict(adapter_checkpoint["model_state_dict"])

    cfg.prenet.in_dim = cleaner.extractor.hubert.config.hidden_size
    prenet = Miipher2PreNet(**cfg.prenet).to(device).eval()

    vocoder_ckpt_path = pathlib.Path(cfg.vocoder_ckpt)
    vocoder_config_path = vocoder_ckpt_path.parent / "config.json"
    if not vocoder_config_path.exists():
        msg = f"Vocoder config file not found at: {vocoder_config_path}"
        raise FileNotFoundError(msg)
    with vocoder_config_path.open() as f:
        h = AttrDict(json.load(f))
    generator = Generator(h).to(device).eval()

    vocoder_checkpoint = torch.load(vocoder_ckpt_path, map_location=device, weights_only=False)
    prenet.load_state_dict(vocoder_checkpoint["prenet"])
    generator.load_state_dict(vocoder_checkpoint["generator"])
    print("Models loaded successfully.")

    # 2. 入力ファイルリストを作成
    input_dir = pathlib.Path(cfg.input_dir)
    output_dir = pathlib.Path(cfg.output_dir)

    if not input_dir.is_dir():
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    audio_files = []
    for ext in cfg.extensions:
        # rglobでサブディレクトリも再帰的に検索
        audio_files.extend(input_dir.rglob(f"*{ext}"))

    if not audio_files:
        print(f"No audio files found in '{input_dir}' with extensions {cfg.extensions}")
        return

    print(f"Found {len(audio_files)} files. Starting batch inference...")

    # 3. 各ファイルに対してループ処理を実行
    for input_path in tqdm(audio_files, desc="Processing files"):
        try:
            # 出力パスを決定 (入力ディレクトリの構造を維持します)
            relative_path = input_path.relative_to(input_dir)
            output_path = output_dir / relative_path

            # 出力ディレクトリを作成
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 3a. 音声ファイルを読み込み
            input_wav = load(input_path).to(device)

            # 3b. 推論実行
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                cleaned_features = cleaner(input_wav)
                mel_spectrogram = prenet(cleaned_features)
                restored_wav = generator(mel_spectrogram)

            # 3c. 修復された音声を保存
            save(output_path, restored_wav.squeeze(0).cpu().to(torch.float32), sr=cfg.output_sampling_rate)

        except Exception as e:
            # エラーが発生しても処理を止めず、エラーメッセージを表示して次のファイルへ進みます
            tqdm.write(f"Failed to process {input_path}: {e}")

    print("\nBatch inference finished.")
