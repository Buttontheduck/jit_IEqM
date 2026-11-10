import wandb
from collections import defaultdict
import numpy as np
import torch


def _mean(values):
    if isinstance(values[0], torch.Tensor):
        return torch.stack(values).float().mean().item()
    return float(np.mean(values))

class MetricsLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsLogger, cls).__new__(cls)
            # Buffer scalar metrics until the next logging interval.
            cls._instance.buffer = defaultdict(list)
        return cls._instance

    def log(self, key: str, value):
        """Add a scalar to the buffer."""
        if isinstance(value, torch.Tensor):
            value = value.detach()
        self.buffer[key].append(value)

    def dump(self, step: int):
        """Average the buffered metrics, push to wandb, and clear the buffer."""
        if not self.buffer:
            return

        # Average on the tensor's device before reading back one number per metric.
        aggregated_metrics = {
            key: _mean(values) for key, values in self.buffer.items()
        }

        # Single wandb call ensures perfectly synchronized steps
        wandb.log(aggregated_metrics, step=step)

        # Start a new logging interval.
        self.buffer.clear()

# Instantiate it here so other files can import this exact instance
global_logger = MetricsLogger()
