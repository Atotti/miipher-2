import argparse
import pathlib

import torch
from omegaconf import OmegaConf

from miipher_2.hifigan.models import Generator
from miipher_2.model.feature_cleaner import FeatureCleaner
from miipher_2.utils.audio import load, save


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=pathlib.Path, required=True)
    ap.add_argument("--vocoder", type=pathlib.Path, required=True)
    ap.add_argument("--in", dest="wav_in", type=pathlib.Path, required=True)
    ap.add_argument("--out", dest="wav_out", type=pathlib.Path, required=True)
    ap.add_argument("--hubert-layer", type=int, default=9, help="HuBERT layer to use (0-based)")
    ap.add_argument("--hubert-model", type=str, default="facebook/hubert-base-ls960", help="HuBERT model name")
    ap.add_argument("--adapter-hidden-dim", type=int, default=1024, help="Adapter hidden dimension")
    return ap.parse_args()


def main() -> None:
    a = parse()

    # FeatureCleaner用の設定オブジェクトを作成
    model_config = OmegaConf.create({
        "hubert_model_name": a.hubert_model,
        "hubert_layer": a.hubert_layer,
        "adapter_hidden_dim": a.adapter_hidden_dim,
    })

    # FeatureCleanerをDictConfigで初期化
    cleaner = FeatureCleaner(model_config).cuda().eval()

    # Accelerateのチェックポイント形式に対応
    try:
        # Accelerateのsave_state形式を試す
        adapter_checkpoint = torch.load(a.adapter, map_location="cpu")
        if "model" in adapter_checkpoint:
            cleaner.load_state_dict(adapter_checkpoint["model"])
        else:
            cleaner.load_state_dict(adapter_checkpoint)
    except Exception as e:
        print(f"Failed to load adapter checkpoint: {e}")
        return

    # HiFi-GANのGenerator初期化（設定ファイルから読み込み）
    vocoder_dir = a.vocoder.parent
    config_path = vocoder_dir / "config.json"

    if config_path.exists():
        import json
        with config_path.open() as f:
            h = json.load(f)
        gen = Generator(h).cuda().eval()
    else:
        # デフォルト設定でGenerator初期化
        from types import SimpleNamespace
        h = SimpleNamespace()
        h.upsample_rates = [8, 8, 2, 2]
        h.upsample_kernel_sizes = [16, 16, 4, 4]
        h.upsample_initial_channel = 512
        h.resblock_kernel_sizes = [3, 7, 11]
        h.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        gen = Generator(h).cuda().eval()

    # Vocoder チェックポイントの読み込み（複数形式に対応）
    try:
        vocoder_checkpoint = torch.load(a.vocoder, map_location="cpu")

        # 形式1: Accelerateのsave_state形式
        if "generator" in vocoder_checkpoint:
            gen.load_state_dict(vocoder_checkpoint["generator"])
        # 形式2: 従来の{"gen": ...}形式
        elif "gen" in vocoder_checkpoint:
            gen.load_state_dict(vocoder_checkpoint["gen"])
        # 形式3: 直接state_dict
        else:
            gen.load_state_dict(vocoder_checkpoint)

    except Exception as e:
        print(f"Failed to load vocoder checkpoint: {e}")
        return

    # 推論実行
    wav = load(a.wav_in).cuda()  # 入力は16kHzにリサンプル
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        feat = cleaner(wav)
        restored = gen(feat)  # 出力は22.05kHz

    a.wav_out.parent.mkdir(parents=True, exist_ok=True)
    # 保存時にサンプリングレートを22050に指定
    save(a.wav_out, restored.cpu(), sr=22050)
    print(f"Restored audio saved to: {a.wav_out}")


if __name__ == "__main__":
    main()
