"""DictaBERT-large-char-menaked encoder for Hebrew G2P.

Loads dicta-il/dictabert-large-char-menaked from Hugging Face and returns the
bare BERT encoder body, without the nikud prediction heads.
"""

from __future__ import annotations

from transformers import AutoModel

MODEL_NAME = "dicta-il/dictabert-large-char-menaked"


def build_encoder(flash_attention: bool = False):
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    return model.bert
