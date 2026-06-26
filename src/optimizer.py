"""Optimizer and LR schedule for Hebrew G2P training."""

from __future__ import annotations

import torch
from transformers import get_cosine_schedule_with_warmup

from model import G2PModel


def build_optimizer(model: G2PModel, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW for the trainable context module and classification heads."""
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"}

    def is_no_decay(name: str) -> bool:
        return any(term in name for term in no_decay)

    return torch.optim.AdamW([
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not is_no_decay(n)],
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and is_no_decay(n)],
            "lr": lr,
            "weight_decay": 0.0,
        },
    ])


def build_scheduler(optimizer: torch.optim.AdamW, warmup_steps: int, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
