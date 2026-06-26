"""Tokenizer for Hebrew G2P - wraps dicta-il/dictabert-large-char-menaked tokenizer."""

from __future__ import annotations

from functools import lru_cache

from tokenizers.pre_tokenizers import Split
from tokenizers import Regex
from transformers import AutoTokenizer, PreTrainedTokenizerFast


MODEL_NAME = "dicta-il/dictabert-large-char-menaked"


def id_to_token(tokenizer) -> dict[int, str]:
    """Invert the tokenizer vocab: token_id -> token string."""
    return {v: k for k, v in tokenizer.get_vocab().items()}


@lru_cache(maxsize=None)
def load_tokenizer() -> PreTrainedTokenizerFast:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # BertPreTokenizer splits on whitespace before WordPiece, collapsing Hebrew
    # words to [UNK]. Replace with a character-level splitter so each character
    # gets its own token with correct offset_mapping.
    tok.backend_tokenizer.pre_tokenizer = Split(
        pattern=Regex("[\\s\\S]"), behavior="isolated"
    )
    return tok
