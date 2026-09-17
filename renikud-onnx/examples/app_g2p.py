# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gradio>=5.0.0",
#   "renikud-onnx @ git+https://github.com/thewh1teagle/renikud.git#subdirectory=renikud-onnx",
#   "numpy>=1.26.0",
#   "phonemizer-fork>=3.3.2",
#   "espeakng-loader>=0.1.9",
# ]
# ///
"""
Hebrew G2P demo using renikud-onnx.
English words in the input are phonemized via espeak before passing to the Hebrew G2P.

Setup:
    wget https://huggingface.co/renikud/renikud/resolve/main/model.onnx -O renikud.onnx

Usage:
    uv run examples/app_g2p.py
"""

import re
import sys
from pathlib import Path

import gradio as gr
import espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer import phonemize as phonemize_en

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from renikud_onnx import G2P

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

RENIKUD_MODEL = "renikud.onnx"
LATIN_WORD_RE = re.compile(r'[a-zA-Z]+')

g2p = G2P(RENIKUD_MODEL)


def to_phonemes(text: str) -> str:
    """Convert Hebrew text to IPA. Hebrew letters via renikud, Latin words via espeak."""
    if not text.strip():
        return ""

    def replace_latin(m: re.Match) -> str:
        return phonemize_en(m.group(0), backend="espeak", language="en-us", strip=True, with_stress=True).strip()

    return g2p.phonemize(LATIN_WORD_RE.sub(replace_latin, text))


demo = gr.Interface(
    fn=to_phonemes,
    inputs=gr.Textbox(label="Hebrew text", placeholder="הקלד טקסט בעברית...", lines=4, rtl=True),
    outputs=gr.Textbox(label="IPA phonemes", lines=3),
    title="Hebrew G2P",
    examples=[
        ["שלום עולם"],
        ["הוא צפה בסרט וראה חיה שצפה במים"],
        ["ראיתי את זה בוואטסאפ של חבר שלי"],
        ["הוא עובד ב Google ומשתמש ב Python כל יום"],
    ],
)

if __name__ == "__main__":
    demo.launch()
