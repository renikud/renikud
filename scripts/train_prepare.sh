#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:?"Usage: $0 <input.tsv>"}

uv run scripts/split_dataset.py "$INPUT" data/train.tsv data/val.tsv

uv run scripts/prepare_align.py data/train.tsv data/train_alignment.jsonl
uv run scripts/prepare_align.py data/val.tsv data/val_alignment.jsonl

uv run scripts/prepare_tokens.py data/train_alignment.jsonl data/.cache/train
uv run scripts/prepare_tokens.py data/val_alignment.jsonl data/.cache/val
