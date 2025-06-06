from peft import LoraConfig, TaskType, get_peft_model
from transformers import HubertModel, Trainer, TrainingArguments

# from miipher_2.dataset import SpeechRestoreWDS # SpeechRestoreWDSはこのファイル内で定義
# from miipher_2.utils import DataCollatorAudioPad # DataCollatorAudioPadは別途定義が必要と仮定

import io
import torch
import torchaudio
import webdataset as wds
from torch.utils.data import IterableDataset
import itertools

# --- LoRA設定 ---
LORA_RANK = 8
LORA_ALPHA = 16
DROP = 0.05

lora_cfg = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=DROP,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "out_proj", # Self-Attention
        "fc1", "fc2", # Feed-Forward
    ],
)

# --- モデルのロード ---
base_model: HubertModel = HubertModel.from_pretrained("rinna/japanese-hubert-large")
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()

# --- SpeechRestoreWDS データセットクラスの定義 ---
def decode_audio_from_bytes(key: str, data: bytes):
    """指定されたキーとバイトデータからオーディオテンソルをデコードします。"""
    try:
        waveform, sr = torchaudio.load(io.BytesIO(data))
        # グローバル変数 target_sample_rate を参照するため、関数内で定義するか引数で渡すことを推奨
        # if sr != target_sample_rate: # target_sample_rate がこのスコープで利用可能である必要あり
        #     waveform = torchaudio.functional.resample(waveform, sr, target_sample_rate)
        return waveform.squeeze(0) # (1, T) -> (T) または (C, T) -> (T) (モノラルを想定)
    except Exception as e:
        print(f"Error decoding audio for key '{key}': {e}")
        return None

class SpeechRestoreWDS(IterableDataset):
    def __init__(self, processed_pipeline: wds.DataPipeline, sr: int = 16_000):
        super().__init__()
        self.processed_pipeline = processed_pipeline
        self.sr = sr

    def __iter__(self):
        for item in self.processed_pipeline:
            if item is None or not isinstance(item, (tuple, list)) or len(item) != 2 or \
               item[0] is None or item[1] is None:
                continue
            degraded, clean = item
            yield {
                "input_values": degraded,
                "labels": clean,
            }

# --- データセットパラメータ ---
tar_pattern = "/home/ayu/datasets/jvs/jvs-train-%06d.tar.gz" # ご自身のパスに修正してください
target_sample_rate = 16_000
num_eval_samples = 1000 # 評価に使用するサンプル数
# JVSデータセットの 'nonpara30' と 'parallel100' の音声ファイル総数は約14982（ユーザー提供の過去のログより）
# ただし、TARファイル化の際にどのように分割・格納されているか不明なため、あくまで目安
# ここでは仮に1つのTARファイルに約2500サンプルあると仮定し、6つのTARファイル (000000-000005) を想定
# 全体のサンプル数 estimation: 2500 * 6 = 15000 (これは非常に大まかな仮定)
# 訓練サンプル数 estimation: 15000 - 1000 = 14000
estimated_train_samples = 14000 # この値を実際のデータセットに合わせて調整してください
train_shuffle_buffer = 5000

# --- データ処理パイプライン構築関数 ---
def build_audio_processing_pipeline(source_dataset: wds.WebDataset) -> wds.DataPipeline:
    """
    WebDatasetソースからオーディオをデコードし、前処理を行う共通パイプライン。
    TARファイル内には "degraded_speech.wav" と "speech.wav" という名前の
    ファイルが各サンプルに含まれていることを想定。
    """
    return (
        source_dataset
        .decode(
            wds.handle_extension("wav", decode_audio_from_bytes),
            handler=wds.warn_and_continue
        )
        .to_tuple("degraded_speech.wav", "speech.wav")
        .map_tuple(
            lambda x: x.float() if isinstance(x, torch.Tensor) else None,
            lambda y: y.float() if isinstance(y, torch.Tensor) else None
        )
        .select(lambda x: x is not None and \
                          isinstance(x, (tuple, list)) and \
                          len(x) == 2 and \
                          isinstance(x[0], torch.Tensor) and \
                          isinstance(x[1], torch.Tensor)
        )
    )

# --- 評価データセットの構築 ---
eval_wds_source = wds.WebDataset(
    tar_pattern,
    resampled=False,
    nodesplitter=wds.shardlists.single_node_only,
    shardshuffle=False # shardshuffle警告を抑制
)
eval_base_pipeline = build_audio_processing_pipeline(eval_wds_source)
eval_iterable_taken = itertools.islice(eval_base_pipeline, num_eval_samples)
eval_pipeline_final = wds.DataPipeline(eval_iterable_taken) # DataPipelineでラップ
eval_set = SpeechRestoreWDS(eval_pipeline_final, sr=target_sample_rate)

# --- 訓練データセットの構築 ---
train_wds_source = wds.WebDataset(
    tar_pattern,
    resampled=False,
    nodesplitter=wds.shardlists.single_node_only,
    shardshuffle=False # shardshuffle警告を抑制
)
train_base_pipeline = build_audio_processing_pipeline(train_wds_source)
train_iterable_skipped = itertools.islice(train_base_pipeline, num_eval_samples, None)
train_pipeline_final = wds.DataPipeline(
    train_iterable_skipped,
    wds.filters.shuffle(train_shuffle_buffer)
)
train_set = SpeechRestoreWDS(train_pipeline_final, sr=target_sample_rate)

# --- DataCollator (miipher_2.utils.DataCollatorAudioPad を想定) ---
class DataCollatorAudioPad:
    def __init__(self, padding_value=0.0, return_attention_mask=True):
        self.padding_value = padding_value
        self.return_attention_mask = return_attention_mask

    def __call__(self, features: list[dict[str, torch.Tensor]]):
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]

        input_values_padded = torch.nn.utils.rnn.pad_sequence(
            input_values, batch_first=True, padding_value=self.padding_value
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.padding_value
        )

        batch = {
            "input_values": input_values_padded,
            "labels": labels_padded,
        }
        if self.return_attention_mask:
            attention_mask = torch.zeros_like(input_values_padded, dtype=torch.long)
            for i, seq in enumerate(input_values):
                attention_mask[i, :len(seq)] = 1
            batch["attention_mask"] = attention_mask
        return batch

# --- 学習引数 ---
# max_steps の計算
# gradient_accumulation_steps は TrainingArguments のデフォルトが1なので、ここでは1と仮定
# num_gpus も1と仮定
# もしこれらの値が異なる場合は、計算を調整してください。
per_device_train_batch_size = 8
num_train_epochs = 20
gradient_accumulation_steps = 1 # TrainingArgumentsのデフォルト値
num_gpus = 1 # シングルGPUを仮定

# 訓練サンプル数が不明な場合、max_steps を固定値で設定するか、
# 1エポックあたりのステップ数に基づいて計算します。
# ここでは estimated_train_samples を使って計算します。
if estimated_train_samples > 0:
    steps_per_epoch = estimated_train_samples // (per_device_train_batch_size * gradient_accumulation_steps * num_gpus)
    calculated_max_steps = steps_per_epoch * num_train_epochs
else:
    # 訓練サンプル数が不明な場合は、十分大きな値を設定するか、エラーにする
    # ここでは仮に20000ステップとします。実際のデータセットに合わせて調整してください。
    print("Warning: estimated_train_samples is not set or is zero. Setting max_steps to a default value (20000).")
    print("Please adjust max_steps based on your dataset size and training goals.")
    calculated_max_steps = 20000


args = TrainingArguments(
    output_dir="hubert_lora_restore_output",
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=8, # 評価バッチサイズも変数化可能
    learning_rate=3e-4,
    num_train_epochs=num_train_epochs, # max_steps を指定する場合、これは参考値になる
    logging_steps=100,
    fp16=True,
    gradient_checkpointing=True,
    save_strategy="epoch",
    # evaluation_strategy="epoch", # この引数がないバージョンを使用中
    max_steps=calculated_max_steps, # <--- 修正点: max_steps を指定
)

# --- Trainer の初期化 ---
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=eval_set,
    data_collator=DataCollatorAudioPad(),
)

# --- 学習の実行 ---
if __name__ == '__main__':
    try:
        print(f"Starting training with max_steps = {calculated_max_steps}...")
        trainer.train()
        print("Training finished.")

        print("Merging LoRA weights...")
        merged_encoder = model.merge_and_unload()
        merged_encoder.save_pretrained("hubert_restore_merged")
        print("Merged model saved to hubert_restore_merged.")

    except FileNotFoundError as e:
        print(f"Error: Could not find TAR files. Please check the tar_pattern: {tar_pattern}")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
