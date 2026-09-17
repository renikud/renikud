# renikud-onnx

Hebrew grapheme-to-phoneme (G2P) inference via ONNX. Converts unvocalized Hebrew text to IPA phonemes.

## Features

- Context-aware Hebrew G2P (text → IPA)
- Letter-level constrained decoding
- Passthrough for non-Hebrew text
- Runs on ONNX Runtime (no PyTorch)
- ~20 MB, real-time inference

## Install

```console
uv pip install renikud-onnx
```

## Usage

The model is fetched from [Hugging Face](https://huggingface.co/renikud/renikud) on first use:

```python
from renikud_onnx import G2P

g2p = G2P()
print(g2p.phonemize("שלום עולם"))
# → ʃalˈom ʔolˈam
```

Or pass a local model path:

```console
wget https://huggingface.co/renikud/renikud/resolve/main/model.onnx -O model.onnx
```

```python
g2p = G2P("model.onnx")
```
