from collections import OrderedDict

import torch
from torch import nn


class EMA:
    """
    Exponential Moving Average for model parameters.
    This class maintains a shadow copy of the model's parameters and updates them
    with a decaying average of the current parameters.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        """
        Args:
            model (nn.Module): The model to apply EMA to.
            decay (float): The decay factor for the moving average.
        """
        self.model = model
        self.decay = decay
        self.shadow = OrderedDict()
        self.backup = OrderedDict()

    def register(self) -> None:
        """Register the EMA parameters by creating a shadow copy."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        print(f"[INFO] Registered {len(self.shadow)} parameters for EMA.")

    def update(self) -> None:
        """Update the shadow parameters with the current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average

    def apply_shadow(self) -> None:
        """Apply the shadow parameters to the model for evaluation."""
        self.backup = OrderedDict()
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self) -> None:
        """Restore the original parameters from the backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = OrderedDict()

    def state_dict(self) -> OrderedDict:
        """Return the state dictionary of the shadow parameters."""
        return self.shadow

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        """Load the shadow parameters from a state dictionary."""
        self.shadow = state_dict
        print("[INFO] Loaded EMA shadow parameters.")
