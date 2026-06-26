# Training

## Commands

### 1. Prepare dataset

```console
./scripts/train_prepare.sh data/data.tsv
```

### 2. Train

```console
./scripts/train_scratch.sh
```

### 3. Fine-tune from a checkpoint

```console
./scripts/train_finetune.sh runs/g2p-classifier/checkpoint-5000
```

## Upload Checkpoint to HuggingFace

```console
./scripts/ckpt_upload.sh runs/g2p-classifier/checkpoint-5000
```

## Export to ONNX

From the repository root, pass a checkpoint directory (paths relative to the repo root). An optional second argument sets the output filename (default `model.onnx`, written under `renikud-onnx/`).

```console
./scripts/ckpt_export.sh runs/g2p-augmented/checkpoint-1500
./scripts/ckpt_export.sh runs/g2p-classifier/checkpoint-5000
./scripts/ckpt_export.sh runs/g2p-augmented/checkpoint-1500 my-model.onnx
```

The script wraps `renikud-onnx/scripts/export.py`. Vocabulary and related metadata are embedded in the `.onnx` file, so no extra files are needed at inference time.

## Benchmark

Run the Hebrew G2P benchmark against a checkpoint. If `gt.tsv` is missing in the repo root, the script downloads it from [heb-g2p-benchmark](https://github.com/thewh1teagle/heb-g2p-benchmark).

```console
./scripts/train_bench.sh runs/g2p-classifier/checkpoint-5000
```

Optional: `./scripts/train_bench.sh runs/g2p-classifier/checkpoint-5000 --save report.txt`

## Download Checkpoint

```console
./scripts/ckpt_download.sh                  # downloads to ./checkpoint
```

To fine-tune from a downloaded checkpoint:

```console
./scripts/ckpt_download.sh checkpoint
./scripts/train_finetune.sh checkpoint
```

## CUDA Version

Install PyTorch for your CUDA version using extras:

```console
uv sync --extra cu130  # CUDA 13.0
uv sync --extra cu128  # CUDA 12.8
```

## Flash Attention

DictaBERT is loaded through Hugging Face Transformers. If the active encoder/backend supports Flash Attention, enable it with `--flash-attention`:

Install a compatible prebuilt wheel first:

- **x86_64**: https://github.com/mjun0812/flash-attention-prebuild-wheels
- **aarch64 (ARM)**: https://pypi.jetson-ai-lab.io/sbsa/cu130

```console
./scripts/train_scratch.sh --flash-attention
```

Validate:

```console
uv run python -c "import flash_attn; print(flash_attn.__version__)"
```

## Learning Rates

- `--lr 1e-4` — default learning rate for the trainable context stack and classification heads
- The DictaBERT encoder is frozen and excluded from the optimizer.

## Data Format

Input TSV: `hebrew_text<TAB>ipa_text` — one sentence per line, no header. Hebrew side may have nikud (diacritics are stripped automatically by the aligner).

The aligner outputs JSONL where each line is `{"hebrew sentence": [["char", "ipa_chunk"], ...]}`. Failed alignments are saved to `<output>_failures.txt`.
