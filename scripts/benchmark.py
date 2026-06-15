"""
Benchmark the Hebrew G2P classifier model against ground truth phonemes.

Download benchmark data first:
    wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/web/data/gt.tsv

Usage:
    uv run scripts/benchmark_classifier.py --checkpoint runs/g2p-classifier/checkpoint-5000 --gt gt.tsv
"""

import argparse
import csv
from pathlib import Path

import torch
import jiwer
from tqdm import tqdm
from model import G2PModel
from infer import load_checkpoint, phonemize
from tokenization import load_tokenizer
from constants import MAX_LEN

PUNCT = str.maketrans("", "", ".,?!")


def load_gt(filepath: str):
    data = []
    with open(filepath, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t", fieldnames=["sentence", "phonemes"]):
            data.append({"sentence": row["sentence"], "phonemes": row["phonemes"]})
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gt", type=str, default="gt.tsv")
    parser.add_argument("--ignore-punct", action="store_true")
    parser.add_argument("--save", type=str, default=None, help="Save report to file")
    args = parser.parse_args()

    if not Path(args.gt).exists():
        print(f"Error: {args.gt} not found. Download with:")
        print("wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/web/data/gt.tsv")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer()
    model = G2PModel()
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    gt_data = load_gt(args.gt)
    refs, hyps, results = [], [], []

    for item in tqdm(gt_data, desc="Benchmarking"):
        pred = phonemize(item["sentence"], model, tokenizer, device, MAX_LEN)
        ref = item["phonemes"]
        if args.ignore_punct:
            ref = ref.translate(PUNCT)
            pred = pred.translate(PUNCT)
        refs.append(ref)
        hyps.append(pred)
        results.append({"sentence": item["sentence"], "gt": ref, "pred": pred, "correct": ref == pred})

    cer = jiwer.cer(refs, hyps)
    wer = jiwer.wer(refs, hyps)
    acc = 1 - wer

    print("\nSample Predictions (first 5):")
    for i, r in enumerate(results[:5], 1):
        print(f"\n{i}. Input: {r['sentence']}")
        print(f"   GT:    {r['gt']}")
        print(f"   Pred:  {r['pred']}")

    print(f"\nResults ({len(gt_data)} samples):")
    print(f"  CER: {cer:.4f}")
    print(f"  WER: {wer:.4f}")
    print(f"  Acc: {acc:.1%}")

    if args.save:
        wrong = [r for r in results if not r["correct"]]
        correct = [r for r in results if r["correct"]]
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(f"Results: {len(gt_data)} samples | Acc: {acc:.1%} | CER: {cer:.4f} | WER: {wer:.4f}\n")
            f.write(f"Wrong: {len(wrong)} | Correct: {len(correct)}\n")
            f.write("=" * 80 + "\n\n")
            for r in wrong:
                f.write(f"[WRONG] {r['sentence']}\n")
                f.write(f"  GT:   {r['gt']}\n")
                f.write(f"  PRED: {r['pred']}\n\n")
            f.write("=" * 80 + "\n\n")
            for r in correct:
                f.write(f"[OK] {r['sentence']}\n")
                f.write(f"  {r['gt']}\n\n")
        print(f"Report saved to {args.save}")


if __name__ == "__main__":
    main()
