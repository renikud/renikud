"""DictaBERT-large-char encoder for Hebrew G2P.

Loads dicta-il/dictabert-large-char from HuggingFace and returns the bare
encoder (BERT body only, no MLM head).
"""

from __future__ import annotations

from transformers import AutoModel


def build_encoder(flash_attention: bool = False):
    return AutoModel.from_pretrained("dicta-il/dictabert-large-char")
