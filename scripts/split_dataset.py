"""
Split a TSV dataset into train and val sets.

Usage:
    uv run scripts/split_dataset.py data/data.tsv data/train.tsv data/val.tsv
"""

import argparse
import random
from pathlib import Path

from tqdm import tqdm


def write_split(path: Path, data: list[str], label: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in tqdm(data, desc=label, unit="lines"):
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("train", type=Path)
    parser.add_argument("val", type=Path)
    parser.add_argument("--val-max", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total = sum(1 for _ in args.input.open(encoding="utf-8"))

    with open(args.input, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in tqdm(f, total=total, desc="Reading", unit="lines") if l.strip()]

    random.seed(args.seed)
    random.shuffle(lines)

    val_size = min(args.val_max, len(lines) // 2)
    train_lines = lines[val_size:]
    val_lines = lines[:val_size]

    write_split(args.train, train_lines, "Writing train")
    write_split(args.val, val_lines, "Writing val")

    print(f"Total: {len(lines)} | Train: {len(train_lines)} | Val: {len(val_lines)}")


if __name__ == "__main__":
    main()
