"""CLI argument parsing for training."""

from __future__ import annotations

import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Hebrew G2P classifier model")
    parser.add_argument("--train-dataset", type=str, default="data/.cache/train")
    parser.add_argument("--eval-dataset", type=str, default="data/.cache/val")
    parser.add_argument("--output-dir", type=str, default="runs/g2p-classifier")
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-seconds", type=float, default=300.0, help="Seconds between timed evaluations")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Stop after this many evals without WER improvement; 0 disables")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reset-steps", action="store_true", default=False, help="Load weights from checkpoint but reset step counter and scheduler (for finetuning on new data)")
    parser.add_argument("--unfreeze-encoder-layers", type=int, default=0, help="Train the top N encoder layers while keeping the rest frozen")
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
    )
    parser.add_argument("--save-last", action="store_true", default=True, help="Overwrite output_dir/last at every eval")
    parser.add_argument("--save-best-cer", action="store_true", default=True, help="Overwrite output_dir/best_cer when eval CER improves")
    parser.add_argument("--save-best-wer", action="store_true", default=True, help="Overwrite output_dir/best_wer when eval WER improves")
    parser.add_argument("--save-best-loss", action="store_true", default=True, help="Overwrite output_dir/best_loss when eval loss improves")
    parser.add_argument("--flash-attention", action="store_true", default=False)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    return parser.parse_args()
