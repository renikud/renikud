"""Small ModernBERT encoder config for Renikud distillation.

This is a randomly initialized ~30M parameter encoder that keeps the
DictaBERT character vocabulary interface, so it can be distilled from the
current DictaBERT-large-char teacher without changing tokenization.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Regex
from tokenizers.pre_tokenizers import Split
from transformers import AutoTokenizer
from transformers import ModernBertConfig, ModernBertModel

from constants import MAX_LEN


DICTABERT_CHAR_TOKENIZER = "models/renikud-dictabert-large-char"
DICTABERT_CHAR_MODEL_ID = "dicta-il/dictabert-large-char"


def load_dictabert_char_tokenizer(tokenizer_name_or_path: str = DICTABERT_CHAR_TOKENIZER):
    """Load the DictaBERT char tokenizer, preferring the local checkpoint copy."""
    if Path(tokenizer_name_or_path).exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(DICTABERT_CHAR_MODEL_ID)

    tokenizer.backend_tokenizer.pre_tokenizer = Split(
        pattern=Regex("[\\s\\S]"), behavior="isolated"
    )
    return tokenizer


def build_modernbert_config(
    *,
    tokenizer_name_or_path: str = DICTABERT_CHAR_TOKENIZER,
    max_position_embeddings: int = MAX_LEN,
    hidden_size: int = 448,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 7,
    intermediate_size: int = 1344,
    local_attention: int = 128,
    global_attn_every_n_layers: int = 3,
) -> ModernBertConfig:
    """Return a ~30M ModernBERT encoder config for char-level Hebrew."""
    tokenizer = load_dictabert_char_tokenizer(tokenizer_name_or_path)

    return ModernBertConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=max_position_embeddings,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        local_attention=local_attention,
        global_attn_every_n_layers=global_attn_every_n_layers,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        mlp_dropout=0.0,
        hidden_activation="gelu",
        classifier_pooling="cls",
    )


def build_encoder(config: ModernBertConfig | None = None) -> ModernBertModel:
    """Build a randomly initialized ModernBERT encoder."""
    return ModernBertModel(config or build_modernbert_config())


def count_parameters(model: ModernBertModel) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


if __name__ == "__main__":
    encoder = build_encoder()
    print(f"modernbert_encoder_params={count_parameters(encoder):,}")
    print(encoder.config)
