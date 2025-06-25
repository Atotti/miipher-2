import io
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor

import hydra
import torch
import torchaudio
import tqdm
import webdataset
from omegaconf import DictConfig

from miipher_2.preprocess import DegradationApplier

# --- 並列処理のためのワーカー関数 ---

# 各ワーカープロセスで共有されるグローバル変数
DEGRADATION_APPLIEffR: DegradationApplier | None = None
SAMPLING_RATE: int | None = None
N_REPEATS: int | None = None
FEATURE_CACHE_DIR: pathlib.Path | None = None

def init_worker(cfg: DictConfig) -> None:
    """ワーカープロセスの初期化関数。プロセス起動時に一度だけ呼ばれる。"""
    global DEGRADATION_APPLIER, SAMPLING_RATE, N_REPEATS, FEATURE_CACHE_DIR

    # 劣化モデルを初期化
    DEGRADATION_APPLIER = DegradationApplier(cfg.preprocess.degradation)

    # その他の設定をグローバル変数に保存
    SAMPLING_RATE = cfg.sampling_rate
    N_REPEATS = cfg.preprocess.n_repeats

    # 特徴量キャッシュディレクトリのパスを解決
    if cfg.preprocess.get("feature_cache_dir"):
        FEATURE_CACHE_DIR = pathlib.Path(cfg.preprocess.feature_cache_dir)

    print(f"[Worker {os.getpid()}] Initialized. Cache dir: {FEATURE_CACHE_DIR}")

def process_item_worker(item: dict) -> list[dict[str, bytes | str]] | None:
    """
    単一のデータアイテムを処理するワーカー関数。
    音声の劣化処理と、キャッシュされたHuBERT特徴量の読み込みを行う。
    """
    global DEGRADATION_APPLIER, SAMPLING_RATE, N_REPEATS, FEATURE_CACHE_DIR
    if DEGRADATION_APPLIER is None or SAMPLING_RATE is None or N_REPEATS is None:
        return None

    try:
        basename = item["basename"]
        audio_file_path = pathlib.Path(item["wav_path"])

        # --------------------------------------------------
        # 1. 音声データの処理 (劣化シミュレーション)
        # --------------------------------------------------
        orig_waveform, orig_sample_rate = torchaudio.load(audio_file_path)
        waveform: torch.Tensor = torchaudio.functional.resample(
            orig_waveform, orig_sample_rate, new_freq=SAMPLING_RATE
        )[0]

        with audio_file_path.open(mode="rb") as f:
            wav_bytes = f.read()

        # --------------------------------------------------
        # 2. HuBERT特徴量キャッシュの読み込み
        # --------------------------------------------------
        hubert_feat_bytes = None
        if FEATURE_CACHE_DIR and FEATURE_CACHE_DIR.exists():
            feat_path = FEATURE_CACHE_DIR / f"{basename}.pt"
            if feat_path.exists():
                with feat_path.open("rb") as f:
                    hubert_feat_bytes = f.read()
            else:
                # ファイルが見つからない場合は警告を出す
                print(f"[Worker {os.getpid()}] WARNING: Feature cache not found for {basename} at {feat_path}")

        # --------------------------------------------------
        # 3. 複数回の劣化処理とサンプル作成
        # --------------------------------------------------
        samples: list[dict[str, bytes | str]] = []
        for i in range(N_REPEATS):
            # DegradationApplierを使って音声にノイズを適用
            degraded_speech = DEGRADATION_APPLIER.process(waveform.clone(), SAMPLING_RATE)
            buff = io.BytesIO()
            torchaudio.save(buff, src=degraded_speech.unsqueeze(0), sample_rate=SAMPLING_RATE, format="wav")
            buff.seek(0)
            degraded_bytes = buff.read()

            sample = {
                "__key__": basename + f"_{i}",
                "speech.wav": wav_bytes,
                "degraded_speech.wav": degraded_bytes,
            }

            # 読み込んだ特徴量キャッシュをサンプルに追加
            if hubert_feat_bytes:
                sample["hubert.pt"] = hubert_feat_bytes

            samples.append(sample)

        return samples

    except Exception as e:
        print(f"[Worker {os.getpid()}] ERROR processing {item.get('basename', 'unknown')}: {e}")
        return None


class Preprocessor:
    """
    Preprocess dataset using parallel processing.
    """
    def __init__(self, cfg: DictConfig) -> None:
        """
        Args:
            cfg: hydra config
        """
        self.cfg = cfg
        # データセットのインスタンス化 (e.g., JVSCorpus, ConcatDataset)
        self.dataset = hydra.utils.instantiate(cfg.preprocess.preprocess_dataset)
        print(f"Dataset loaded with {len(self.dataset)} items.")

    def build_from_path(self) -> None:
        """
        データセット全体の前処理を並列で実行し、webdataset形式で保存する。
        """
        output_dir = pathlib.Path(self.cfg.preprocess.train_tar_sink.pattern).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        train_sink = hydra.utils.instantiate(self.cfg.preprocess.train_tar_sink)
        val_sink = hydra.utils.instantiate(self.cfg.preprocess.val_tar_sink)

        num_workers = os.cpu_count() or 1
        print(f"Using {num_workers} workers for preprocessing.")

        # データセットを訓練用と検証用に分割
        val_size = self.cfg.preprocess.val_size
        all_items = list(self.dataset)
        train_items = all_items[val_size:]
        val_items = all_items[:val_size]

        # ProcessPoolExecutorで並列処理を実行
        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(self.cfg,)) as executor:
            # --- 検証データの処理 ---
            print("Processing validation data...")
            # executor.mapは遅延評価なのでlist()でラップして即時実行＆結果を待つ
            val_results = list(tqdm.tqdm(executor.map(process_item_worker, val_items), total=len(val_items)))
            for result_list in val_results:
                if result_list:
                    for sample in result_list:
                        val_sink.write(sample)

            # --- 訓練データの処理 ---
            print("Processing training data...")
            train_results = list(tqdm.tqdm(executor.map(process_item_worker, train_items), total=len(train_items)))
            for result_list in train_results:
                if result_list:
                    for sample in result_list:
                        train_sink.write(sample)

        train_sink.close()
        val_sink.close()
        print("Preprocessing finished successfully.")

