"""DictaBERT-large-char encoder for Hebrew G2P.

Loads dicta-il/dictabert-large-char from Hugging Face and returns the bare
encoder body, without an MLM head.
"""

from __future__ import annotations

from transformers import AutoModel

MODEL_NAME = "dicta-il/dictabert-large-char"


def build_encoder(flash_attention: bool = False):
    kwargs = {}
    if flash_attention:
        kwargs["attn_implementation"] = "sdpa"
    return AutoModel.from_pretrained(MODEL_NAME, **kwargs)
