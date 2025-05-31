from peft import LoraConfig, TaskType, get_peft_model
from transformers import HubertModel

LORA_RANK = 8
LORA_ALPHA = 16
DROP = 0.05

lora_cfg = LoraConfig(
    task_type      = TaskType.FEATURE_EXTRACTION,
    r              = LORA_RANK,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = DROP,
    # HuBERT の線形層名 HF 実装に合わせて列挙
    target_modules = [
        # Self-Attention
        "q_proj", "k_proj", "v_proj", "out_proj",
        # Feed-Forward
        "fc1", "fc2",
    ],
    # conv feature extractor・pos_conv 等は除外
)


base_model: HubertModel = HubertModel.from_pretrained("rinna/japanese-hubert-large")
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()
