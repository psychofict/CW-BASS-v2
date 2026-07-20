import torch
import torch.nn as nn
from copy import deepcopy


class EMATeacher(nn.Module):
    """Exponential Moving Average teacher model.

    Following UniMatch-V2 EMA pattern:
      - Ramp-up: decay = min(1 - 1/(iter+1), max_decay)
      - Updates both parameters AND buffers (BatchNorm stats)

    Args:
        student_model: the student nn.Module to track
        decay: initial/max EMA decay factor (default 0.996)
    """

    def __init__(self, student_model, decay=0.996):
        super().__init__()
        self.decay = decay
        self.model = deepcopy(student_model)
        # Freeze teacher parameters — updated via EMA only
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student_model):
        """Update teacher parameters with EMA of student parameters."""
        d = self.decay
        # Update parameters
        student_params = dict(student_model.named_parameters())
        for name, param in self.model.named_parameters():
            if name in student_params:
                param.data.mul_(d).add_(student_params[name].data, alpha=1.0 - d)
        # Update buffers (BatchNorm running stats)
        student_buffers = dict(student_model.named_buffers())
        for name, buf in self.model.named_buffers():
            if name in student_buffers:
                if buf.dtype.is_floating_point:
                    buf.data.mul_(d).add_(student_buffers[name].data, alpha=1.0 - d)
                else:
                    # Non-float buffers (e.g. num_batches_tracked) — just copy
                    buf.data.copy_(student_buffers[name].data)

    def forward(self, x):
        return self.model(x)

    def state_dict(self, **kwargs):
        return {
            'decay': self.decay,
            'model_state_dict': self.model.state_dict(),
        }

    def load_state_dict(self, state_dict, **kwargs):
        self.decay = state_dict['decay']
        self.model.load_state_dict(state_dict['model_state_dict'])
