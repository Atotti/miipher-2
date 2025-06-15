import torch
import torch.nn.functional as F  # noqa: N812


class DataCollatorAudioPad:
    """
    Pad 1-D waveforms to the max length in batch (zero-padding).
    Returns dict ready for Trainer.
    """

    def __call__(self, features):
        inputs = [f["input_values"] for f in features]
        targets = [f["labels"] for f in features]

        max_len = max(x.size(-1) for x in inputs)
        batch_in = torch.stack([F.pad(x, (0, max_len - x.size(-1))) for x in inputs])
        batch_lbl = torch.stack([F.pad(y, (0, max_len - y.size(-1))) for y in targets])

        return {"input_values": batch_in, "labels": batch_lbl}
