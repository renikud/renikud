# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "jiwer>=4.0.0",
#   "onnxruntime>=1.24.2",
#   "tqdm>=4.0.0",
# ]
# ///
"""
Benchmark renikud-onnx model against ground truth phonemes.

Download benchmark data first:
    wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/web/data/gt.tsv

Usage:
    uv run scripts/benchmark.py --model model.onnx --gt gt.tsv
"""

import argparse
import csv
import sys
from pathlib import Path

import jiwer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from renikud_onnx import G2P

PUNCT = str.maketrans("", "", ".,?!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.onnx")
    parser.add_argument("--gt", default="gt.tsv")
    parser.add_argument("--ignore-punct", action="store_true")
    args = parser.parse_args()

    if not Path(args.gt).exists():
        print(f"Error: {args.gt} not found. Download with:")
        print("wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/web/data/gt.tsv")
        return

    g2p = G2P(args.model)

    gt_data = []
    with open(args.gt) as f:
        for row in csv.DictReader(f, delimiter="\t", fieldnames=["sentence", "phonemes"]):
            gt_data.append({"sentence": row["sentence"], "phonemes": row["phonemes"]})

    refs, hyps, examples = [], [], []
    for item in tqdm(gt_data, desc="Benchmarking"):
        pred = g2p.phonemize(item["sentence"])
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
