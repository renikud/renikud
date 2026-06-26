"""Hebrew G2P classifier model — per-character prediction of consonant, vowel, and stress."""

from __future__ import annotations

import torch
import torch.nn as nn
from constants import IGNORE_INDEX
from encoder import build_encoder
from phonology import NUM_CONSONANT_CLASSES, NUM_VOWEL_CLASSES, NUM_STRESS_CLASSES, build_consonant_mask, apply_consonant_mask


class G2PModel(nn.Module):
    """
    Per-character Hebrew G2P model.

    For each Hebrew letter in the input, predicts:
      - consonant class (from a per-letter constrained set)
      - vowel class     (a / e / i / o / u / ∅)
      - stress          (yes / no)

    Non-Hebrew characters (spaces, punctuation, digits, Latin) are passed
    through unchanged at inference — the heads are never called for them.
    """

    def __init__(self, dropout_rate: float = 0.1, flash_attention: bool = False) -> None:
        super().__init__()

        self.encoder = build_encoder(flash_attention=flash_attention)
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.encoder.eval()

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)
        context_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=16,
            dim_feedforward=2816,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(context_layer, num_layers=2, enable_nested_tensor=False)

        # Coupled classification heads: each head sees encoder state + previous head logits
        self.consonant_head = nn.Linear(hidden_size, NUM_CONSONANT_CLASSES)
        self.vowel_head = nn.Linear(hidden_size + NUM_CONSONANT_CLASSES, NUM_VOWEL_CLASSES)
        self.stress_head = nn.Linear(hidden_size + NUM_CONSONANT_CLASSES + NUM_VOWEL_CLASSES, NUM_STRESS_CLASSES)

        self._consonant_mask: torch.Tensor = build_consonant_mask()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        consonant_labels: torch.Tensor | None = None,
        vowel_labels: torch.Tensor | None = None,
        stress_labels: torch.Tensor | None = None,
        tokenizer_vocab: dict[int, str] | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        hidden = self.dropout(encoder_outputs.last_hidden_state)  # [B, S, H]
        hidden = self.context(hidden, src_key_padding_mask=attention_mask == 0)
        hidden = self.dropout(hidden)

        consonant_logits = self.consonant_head(hidden)  # [B, S, NUM_CONSONANT_CLASSES]

        # Couple heads using raw (unmasked) consonant logits — consistent between train and inference
        vowel_logits = self.vowel_head(torch.cat([hidden, consonant_logits], dim=-1))                          # [B, S, NUM_VOWEL_CLASSES]
        stress_logits = self.stress_head(torch.cat([hidden, consonant_logits, vowel_logits], dim=-1))          # [B, S, NUM_STRESS_CLASSES]

        # Apply consonant mask only to the output (inference constraint, not used during training)
        if tokenizer_vocab is not None:
            consonant_logits = apply_consonant_mask(consonant_logits, input_ids, tokenizer_vocab, self._consonant_mask)

        output: dict[str, torch.Tensor] = {
            "consonant_logits": consonant_logits,
            "vowel_logits": vowel_logits,
            "stress_logits": stress_logits,
        }

        if consonant_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            loss = (
                loss_fct(consonant_logits.view(-1, NUM_CONSONANT_CLASSES), consonant_labels.view(-1))
                + loss_fct(vowel_logits.view(-1, NUM_VOWEL_CLASSES), vowel_labels.view(-1))
                + loss_fct(stress_logits.view(-1, NUM_STRESS_CLASSES), stress_labels.view(-1))
            )
            output["loss"] = loss

        return output
