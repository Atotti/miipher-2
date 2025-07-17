import torch

# 元のチェックポイントファイルを指定
checkpoint_path = 'exp/adapter_layer_6_mhubert_147/checkpoint_199k.pt'

# weights_only=False を追加して読み込む
print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
print("Checkpoint loaded successfully.")

# 新しいstate_dictを作成
model_state_dict = checkpoint['model_state_dict']
new_state_dict = {}

# 'extractor.hubert.' を 'extractor.model.' に置換する
for key, value in model_state_dict.items():
    if key.startswith('extractor.hubert.'):
        new_key = key.replace('extractor.hubert.', 'extractor.model.', 1)
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value # その他のキーはそのままコピー

# 新しいstate_dictをチェックポイントにセットする
checkpoint['model_state_dict'] = new_state_dict

# 修正したチェックポイントを新しいファイルとして保存する
new_checkpoint_path = 'exp/adapter_layer_6_mhubert_147/checkpoint_199k_fixed.pt'
torch.save(checkpoint, new_checkpoint_path)

print(f"Fixed checkpoint saved to: {new_checkpoint_path}")
