# Evaluation

## Metrics

All evaluation uses [jiwer](https://github.com/jitsi/jiwer) to compute:

- **CER** (Character Error Rate) — primary metric, measures phoneme-level accuracy
- **WER** (Word Error Rate) — secondary metric, measures per-word accuracy
- **Acc** (Word Accuracy) — `1 - WER`, used for high-level comparison

### During Training

`src/train.py` computes consonant accuracy, vowel accuracy, and stress accuracy on the validation split at each eval step.

### Standalone Benchmark

`scripts/benchmark.py` evaluates a checkpoint against an external ground-truth file:

```console
uv run scripts/benchmark.py --checkpoint outputs/my-run/checkpoint-1200 --gt gt.tsv
```

Optional `--ignore-punct` strips `.,?!` from both references and predictions before scoring.

## Benchmark Dataset

The benchmark is [heb-g2p-benchmark](https://github.com/thewh1teagle/heb-g2p-benchmark) — 100 Hebrew sentences specifically engineered to stress-test G2P systems. The sentences cover:

- Male vs. female verb forms
- Stress placement variation
- Homographs (words with different readings depending on context)

Download:
```console
wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/web/data/gt.tsv
```

## Results

| Model | Acc | WER | CER |
|---|---|---|---|
| Phonikud (teacher) | 86.1% | 0.14 | 0.04 |
| **Ours (best)** | **89.3%** | 0.107 | 0.026 |

We have surpassed Phonikud — the model now exceeds its own training signal.
