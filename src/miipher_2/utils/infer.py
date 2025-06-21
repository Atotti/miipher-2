import argparse
import os

import torch
from omegaconf import DictConfig, OmegaConf

from ..extractors.hubert import HubertExtractor
from ..hifigan.models import Generator
from ..model.feature_cleaner import FeatureCleaner
from ..utils.checkpoint import load_checkpoint
from .audio import load, save


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-ckpt", type=str, required=True, help="Path to adapter checkpoint")
    parser.add_argument("--vocoder-ckpt", type=str, required=True, help="Path to vocoder checkpoint")
    parser.add_argument("--wav-in", type=str, required=True, help="Input wav file")
    parser.add_argument("--wav-out", type=str, required=True, help="Output wav file")
    parser.add_argument("--hubert-model-name", type=str, default="facebook/hubert-base-ls960", help="HuBERT model name")
    parser.add_argument("--hubert-layer", type=int, default=6, help="HuBERT layer to extract")
    parser.add_argument("--adapter-hidden-dim", type=int, default=1024, help="Adapter hidden dimension")
    return parser.parse_args()


def main() -> None:
    a = parse()

    # ★★★ Checkpointから設定を自動読み取り（レイヤ番号の食い違い防止）
    try:
        adapter_checkpoint = load_checkpoint(a.adapter_ckpt)
        ckpt_cfg = adapter_checkpoint.get("config", {}).get("model", {})

        # Checkpoint内の設定があればそれを優先使用
        hubert_layer = ckpt_cfg.get("hubert_layer", a.hubert_layer)
        hubert_model_name = ckpt_cfg.get("hubert_model_name", a.hubert_model_name)

        if hubert_layer != a.hubert_layer:
            print(f"⚠️  CLI引数 hubert_layer={a.hubert_layer} をCheckpoint設定 {hubert_layer} で上書きします")
        if hubert_model_name != a.hubert_model_name:
            print(
                f"⚠️  CLI引数 hubert_model_name={a.hubert_model_name} をCheckpoint設定 {hubert_model_name} で上書きします"
            )

    except Exception as e:
        print(f"⚠️  Checkpoint設定読み取りに失敗、CLI引数を使用: {e}")
        hubert_layer = a.hubert_layer
        hubert_model_name = a.hubert_model_name

    # HubertExtractorを初期化（共有可能）
    hubert_extractor = HubertExtractor(
        model_name=hubert_model_name,
        layer=hubert_layer,
    )

    # FeatureCleanerを初期化（DictConfigとextractor注入）
    model_cfg = DictConfig(
        {
            "hubert_model_name": hubert_model_name,
            "hubert_layer": hubert_layer,
            "adapter_hidden_dim": 1024,  # デフォルト値
        }
    )
    feature_cleaner = FeatureCleaner(model_cfg, hubert_extractor=hubert_extractor)

    # Accelerateのチェックポイント形式に対応
    try:
        if os.path.isdir(a.adapter_ckpt):
            # Accelerate形式（ディレクトリ）
            adapter_state_path = os.path.join(a.adapter_ckpt, "pytorch_model.bin")
            if os.path.exists(adapter_state_path):
                adapter_state = torch.load(adapter_state_path, map_location="cpu")
                feature_cleaner.load_state_dict(adapter_state)
            else:
                msg = f"pytorch_model.bin not found in {a.adapter_ckpt}"
                raise FileNotFoundError(msg)
        else:
            # 従来形式（単一ファイル）
            adapter_checkpoint = torch.load(a.adapter_ckpt, map_location="cpu")
            if "model_state_dict" in adapter_checkpoint:
                feature_cleaner.load_state_dict(adapter_checkpoint["model_state_dict"])
            elif "model" in adapter_checkpoint:
                feature_cleaner.load_state_dict(adapter_checkpoint["model"])
            else:
                feature_cleaner.load_state_dict(adapter_checkpoint)
    except Exception as e:
        print(f"Failed to load adapter checkpoint: {e}")
        return

    # Vocoder（HiFi-GAN）の読み込み
    try:
        vocoder_checkpoint = torch.load(a.vocoder_ckpt, map_location="cpu")
        # ★★★ 設定の正しいアクセス方法
        if "config" in vocoder_checkpoint:
            gen_cfg = vocoder_checkpoint["config"]["model"]
        else:
            # デフォルト設定
            gen_cfg = {
                "upsample_rates": [8, 8, 2, 2],
                "upsample_kernel_sizes": [16, 16, 4, 4],
                "upsample_initial_channel": 512,
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "resblock": "1",  # ResBlock1を使用
            }

        # SimpleNamespaceでh objectを作成（Generatorの要求形式）
        from types import SimpleNamespace

        h = SimpleNamespace()
        h.upsample_rates = gen_cfg["upsample_rates"]
        h.upsample_kernel_sizes = gen_cfg["upsample_kernel_sizes"]
        h.upsample_initial_channel = gen_cfg["upsample_initial_channel"]
        h.resblock_kernel_sizes = gen_cfg["resblock_kernel_sizes"]
        h.resblock_dilation_sizes = gen_cfg.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        h.resblock = gen_cfg.get("resblock", "1")

        gen = Generator(h)

        if "generator" in vocoder_checkpoint:
            gen.load_state_dict(vocoder_checkpoint["generator"])
        elif "model_state_dict" in vocoder_checkpoint:
            gen.load_state_dict(vocoder_checkpoint["model_state_dict"])
        else:
            gen.load_state_dict(vocoder_checkpoint)

    except Exception as e:
        print(f"Failed to load vocoder checkpoint: {e}")
        return

    # GPU推論
    feature_cleaner = feature_cleaner.cuda().eval()
    gen = gen.cuda().eval()

    # 推論実行
    wav = load(a.wav_in).cuda()  # 入力は16kHzにリサンプル
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        feat = feature_cleaner(wav)
        restored = gen(feat)  # 出力は22.05kHz

    # 結果保存
    save(restored.squeeze().cpu(), a.wav_out, 22050)


if __name__ == "__main__":
    main()
