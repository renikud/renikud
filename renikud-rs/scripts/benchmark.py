# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "jiwer>=4.0.0",
# ]
# ///
"""
Build the Rust phonemizer and benchmark it against ground truth phonemes.

Usage:
    uv run scripts/benchmark.py --model model.onnx --gt ../gt.tsv
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import jiwer

PUNCT = str.maketrans("", "", ".,?!")
REPO_DIR = Path(__file__).parent.parent.parent


def build(release: bool = True) -> Path:
    mode = ["--release"] if release else []
    print("Building Rust binary...")
    subprocess.run(
        ["cargo", "build", "--example", "phonemize"] + mode,
        cwd=REPO_DIR,
        check=True,
    )
    profile = "release" if release else "debug"
    return REPO_DIR / "target" / profile / "examples" / "phonemize"


def run_phonemize(binary: Path, model: str, sentences: list[str]) -> list[str]:
    input_text = "\n".join(sentences) + "\n"
    result = subprocess.run(
        [str(binary), model],
        input=input_text,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.onnx")
    parser.add_argument("--gt", default="../gt.tsv")
    parser.add_argument("--ignore-punct", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Build in debug mode")
    args = parser.parse_args()

    if not Path(args.gt).exists():
        print(f"Error: {args.gt} not found. Download with:")
        print("wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/web/data/gt.tsv")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"Error: {args.model} not found.")
        sys.exit(1)

    gt_data = []
    with open(args.gt, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            gt_data.append({"sentence": row["Sentence"], "phonemes": row["Phonemes"]})

    binary = build(release=not args.debug)
    sentences = [item["sentence"] for item in gt_data]
    preds = run_phonemize(binary, args.model, sentences)

    if len(preds) != len(gt_data):
        print(f"Error: expected {len(gt_data)} predictions, got {len(preds)}")
        sys.exit(1)

    refs, hyps, examples = [], [], []
    for item, pred in zip(gt_data, preds):
        ref = item["phonemes"]
        if args.ignore_punct:
            ref = ref.translate(PUNCT)
            pred = pred.translate(PUNCT)
        refs.append(ref)
        hyps.append(pred)
        if len(examples) < 5:
            examples.append({"sentence": item["sentence"], "gt": item["phonemes"], "pred": pred})

    print("\nSample Predictions (first 5):")
    for i, ex in enumerate(examples, 1):
        print(f"\n{i}. Input: {ex['sentence']}")
        print(f"   GT:    {ex['gt']}")
        print(f"   Pred:  {ex['pred']}")

    print(f"\nResults ({len(gt_data)} samples):")
    print(f"  CER: {jiwer.cer(refs, hyps):.4f}")
    print(f"  WER: {jiwer.wer(refs, hyps):.4f}")
    print(f"  Acc: {1 - jiwer.wer(refs, hyps):.1%}")


if __name__ == "__main__":
    main()
