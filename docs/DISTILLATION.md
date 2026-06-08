# Distillation

Distillation trains a small ModernBERT student to match the current
DictaBERT-large-char ReNikud teacher.

## Teacher

Default teacher checkpoint:

```console
models/renikud-dictabert-large-char
```

This is a 306M parameter DictaBERT-large-char encoder with the ReNikud coupled
heads:

```text
hidden -> consonant_head
hidden + consonant_logits -> vowel_head
hidden + consonant_logits + vowel_logits -> stress_head
```

The student preserves this head topology.

## Student

The ModernBERT student is defined in:

```text
src/modernbert_encoder.py
```

Current default size:

```text
hidden_size: 448
layers: 12
heads: 7
intermediate: 1344
params: ~31.7M including G2P heads
```

It uses the same DictaBERT character tokenizer interface. The tokenizer is
forced to character-level splitting, matching `src/tokenization.py`.

## Losses

`src/train_distill.py` uses:

```text
KL(student consonant logits, teacher consonant logits)
KL(student vowel logits, teacher vowel logits)
KL(student stress logits, teacher stress logits)
hidden-state cosine loss through train-only projection layers
optional CE loss when gold token labels are available
```

Default weights:

```text
consonant KL: 1.0
vowel KL:     2.0
stress KL:    1.5
hidden:       0.5
CE:           0.25, only when labels exist
temperature:  2.0
```

For raw TSV distillation, there are no token labels, so CE is skipped.

## Raw TSV Distillation

The raw Knesset file:

```text
data/knesset_phonemes_v2_no_nikud.txt
```

has format:

```text
Hebrew sentence<TAB>IPA phonemes
```

For pure distillation, only the Hebrew input is used for training. The phoneme
column is ignored during training.

Example launch:

```console
uv run accelerate launch src/train_distill.py \
  --train-tsv data/knesset_phonemes_v2_no_nikud.txt \
  --benchmark-gt gt.tsv \
  --teacher-checkpoint models/renikud-dictabert-large-char \
  --output-dir outputs/g2p-modernbert-distill \
  --epochs 2 \
  --train-batch-size 32 \
  --encoder-lr 5e-5 \
  --head-lr 2e-4 \
  --projector-lr 2e-4 \
  --warmup-steps 500 \
  --logging-steps 50 \
  --save-steps 5000 \
  --dataloader-workers 2 \
  --save-last \
  --save-best-cer \
  --save-best-wer \
  --fp16
```

## Benchmark

Use the same external benchmark as `docs/EVALUATION.md`:

```console
wget -O gt.tsv https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/web/data/gt.tsv
```

During raw distillation, checkpoint selection should use:

```text
best_wer
best_cer
last
```

`best_loss` is not meaningful for raw TSV distillation because the benchmark has
WER/CER but no token-level eval loss.

Raw benchmark eval prints `loss=nan` for this reason. This is expected. The
training loss is still valid and logged separately as `train_loss`; only
benchmark `eval_loss` is unavailable without aligned token labels.

## Metrics

Use losses for diagnostics:

```text
distill KL down = student is matching teacher distributions
hidden loss down = student representations are aligning
```

Use benchmark quality for model selection:

```text
WER down = primary signal
CER down = secondary signal
```

If KL improves but WER does not, reduce hidden weight, add/raise gold CE, or
check decoding/masking.

## Implementation Notes

- Student consonant masking is registered as a buffer and applied only when
  `tokenizer_vocab` is passed, matching inference behavior.
- Training KL uses unmasked logits. Inference and evaluation use constrained
  consonant decoding.
- Hidden projectors are train-only adapters. They are saved as
  `projectors.safetensors` for resume, but are not needed for inference/export.
- Raw TSV batches must contain tensors only; Accelerate cannot concatenate
  string fields from an iterable dataloader.
- Clip gradients once per optimizer step across all trainable params. With AMP,
  calling `accelerator.clip_grad_norm_` twice for the same optimizer can trigger
  `unscale_()` errors.
