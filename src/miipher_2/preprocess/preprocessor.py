import multiprocessing as mp
import io
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor

import hydra
import torch
import torchaudio
import tqdm
from omegaconf import DictConfig

from miipher_2.preprocess import DegradationApplier

# --- 並列処理のためのワーカー関数 (変更なし) ---

DEGRADATION_APPLIER: DegradationApplier | None = None
SAMPLING_RATE: int | None = None
N_REPEATS: int | None = None
FEATURE_CACHE_DIR: pathlib.Path | None = None

DEGRADATION_APPLIER: DegradationApplier | None = None   # ワーカー側参照
DEGRADATION_APPLIER_G: DegradationApplier | None = None # 親が作る実体


def init_worker(cfg: DictConfig) -> None:
    """ワーカープロセスの初期化関数。プロセス起動時に一度だけ呼ばれる。"""
    global DEGRADATION_APPLIER, DEGRADATION_APPLIER_G, SAMPLING_RATE, N_REPEATS, FEATURE_CACHE_DIR
    DEGRADATION_APPLIER = DEGRADATION_APPLIER_G
    SAMPLING_RATE = cfg.sampling_rate
    N_REPEATS = cfg.preprocess.n_repeats
    if cfg.preprocess.get("feature_cache_dir"):
        FEATURE_CACHE_DIR = pathlib.Path(cfg.preprocess.feature_cache_dir)
    print(f"[Worker {os.getpid()}] Initialized. Cache dir: {FEATURE_CACHE_DIR}")


def process_item_worker(item: dict) -> list[dict[str, bytes | str]] | None:
    """単一のデータアイテムを処理するワーカー関数。"""
    global DEGRADATION_APPLIER, SAMPLING_RATE, N_REPEATS, FEATURE_CACHE_DIR
    if DEGRADATION_APPLIER is None or SAMPLING_RATE is None or N_REPEATS is None:
        return None
    try:
        basename = item["basename"]
        audio_file_path = pathlib.Path(item["wav_path"])

        orig_waveform, orig_sample_rate = torchaudio.load(audio_file_path)
        waveform: torch.Tensor = torchaudio.functional.resample(
            orig_waveform, orig_sample_rate, new_freq=SAMPLING_RATE
        )[0]
        with audio_file_path.open(mode="rb") as f:
            wav_bytes = f.read()

        hubert_feat_bytes = None
        if FEATURE_CACHE_DIR and FEATURE_CACHE_DIR.exists():
            feat_path = FEATURE_CACHE_DIR / f"{basename}.pt"
            if feat_path.exists():
                with feat_path.open("rb") as f:
                    hubert_feat_bytes = f.read()
            else:
                print(f"[Worker {os.getpid()}] WARNING: Feature cache not found for {basename} at {feat_path}")

        samples: list[dict[str, bytes | str]] = []
        for i in range(N_REPEATS):
            degraded_speech = DEGRADATION_APPLIER.process(waveform.clone(), SAMPLING_RATE)
            buff = io.BytesIO()
            torchaudio.save(buff, src=degraded_speech.unsqueeze(0), sample_rate=SAMPLING_RATE, format="wav")
            buff.seek(0)

            sample = {
                "__key__": basename + f"_{i}",
                "speech.wav": wav_bytes,
                "degraded_speech.wav": buff.read(),
            }
            if hubert_feat_bytes:
                sample["hubert.pt"] = hubert_feat_bytes
            samples.append(sample)
        return samples
    except Exception as e:
        print(f"[Worker {os.getpid()}] ERROR processing {item.get('basename', 'unknown')}: {e}")
        return None


# --- Preprocessor クラスの修正 ---


class Preprocessor:
    """Preprocess dataset using parallel processing."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.dataset = hydra.utils.instantiate(cfg.preprocess.preprocess_dataset)
        print(f"Dataset loaded with {len(self.dataset)} items.")

    def build_from_path(self) -> None:
        """データセット全体の前処理を並列で実行し、webdataset形式で保存する。"""
        global DEGRADATION_APPLIER_G
        DEGRADATION_APPLIER_G = DegradationApplier(self.cfg.preprocess.degradation)
        output_dir = pathlib.Path(self.cfg.preprocess.train_tar_sink.pattern).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        train_sink = hydra.utils.instantiate(self.cfg.preprocess.train_tar_sink)
        val_sink = hydra.utils.instantiate(self.cfg.preprocess.val_tar_sink)
        num_workers = os.cpu_count() or 1
        print(f"Using {num_workers} workers for preprocessing.")

        val_size = self.cfg.preprocess.val_size
        all_items = list(self.dataset)
        train_items = all_items[val_size:]
        val_items = all_items[:val_size]

        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("fork"), initializer=init_worker, initargs=(self.cfg,)) as executor:
            # --- ここから変更 ---
            # list()で全結果を待つのではなく、ループで結果を一つずつ処理する

            print("Processing validation data...")
            with tqdm.tqdm(total=len(val_items)) as pbar:
                # executor.mapから返されるイテレータを直接ループ処理
                for result_list in executor.map(process_item_worker, val_items, chunksize=32):
                    if result_list:
                        for sample in result_list:
                            val_sink.write(sample)
                    pbar.update(1)  # 1アイテム処理するごとにプログレスバーを更新

            print("Processing training data...")
            with tqdm.tqdm(total=len(train_items)) as pbar:
                for result_list in executor.map(process_item_worker, train_items, chunksize=32):
                    if result_list:
                        for sample in result_list:
                            train_sink.write(sample)
                    pbar.update(1)
            # --- ここまで変更 ---

        train_sink.close()
        val_sink.close()
        print("Preprocessing finished successfully.")
