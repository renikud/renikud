<p align="center">
  <a target="_blank" href="https://renikud.github.io">
    <img
        width="110px"
        alt="ReNikud logo"
        src="./design/logo.webp"
    />
  </a>
</p>

<h1 align="center">ReNikud - Audio-Supervised Hebrew Grapheme-to-Phoneme</h1>

<p align="center">
  <em>Convert unvocalized Hebrew text into IPA for TTS, speech technology, and spoken-language research</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/IEEE_SLT-2026-blue" alt="IEEE SLT 2026" />
  &nbsp;
  <a target="_blank" href="https://renikud.github.io">
    <img src="https://img.shields.io/badge/🌐_Project-Page-green" alt="Project Page" />
  </a>
  &nbsp;
  <a target="_blank" href="https://arxiv.org/abs/2606.20179">
    <img src="https://img.shields.io/badge/arXiv-2606.20179-b31b1b" alt="arXiv" />
  </a>
</p>

<hr />

## Features

- Context-aware Hebrew G2P (text → IPA)
- Letter-level constrained decoding
- Passthrough for non-Hebrew text
- Runs on ONNX Runtime (no PyTorch)
- ~20 MB, real-time inference

## Usage

Inference is published as **`renikud-onnx`** on PyPI. Install and download the ONNX weights from Hugging Face (they are not bundled with the wheel):

```console
pip install renikud-onnx
wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx -O model.onnx
```

```python
from renikud_onnx import G2P

g2p = G2P("model.onnx")
print(g2p.phonemize("שלום עולם"))
# → ʃalˈom ʔolˈam
```

See `renikud-onnx/README.md` for the same install / download / usage flow. For Rust inference, see `renikud-rs/`.

## Architecture

See `docs/ARCHITECTURE.md` for model design and implementation details.

## Training

See `docs/TRAINING.md` for data preparation, training commands, upload/download, ONNX export, benchmark, and hyperparameters.

## License

This project is licensed under the Creative Commons Attribution 4.0 International license (CC BY 4.0). See [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{melichov2026renikud,
  title={ReNikud: Audio-Supervised Hebrew Grapheme-to-Phoneme Conversion},
  author={Maxim Melichov and Yakov Kolani and Morris Alper},
  booktitle={Proc. IEEE SLT 2026},
  year={2026},
  url={https://arxiv.org/pdf/2606.20179},
}
```
