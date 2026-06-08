"""Evaluation helpers for the Hebrew G2P classifier."""

from __future__ import annotations

import torch
import jiwer

from constants import IGNORE_INDEX
from decoder import decode
from tokenization import id_to_token


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Per-token accuracy ignoring IGNORE_INDEX positions."""
    mask = labels != IGNORE_INDEX
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return (preds[mask] == labels[mask]).float().mean().item()


def decode_batch(texts: list[str], out: dict, tokenizer) -> list[str]:
    preds = []
    for i, text in enumerate(texts):
        encoding = tokenizer(text, truncation=True, max_length=512, return_offsets_mapping=True, return_tensors=None)
        preds.append(decode(
            text=text,
            offset_mapping=encoding["offset_mapping"],
            consonant_logits=out["consonant_logits"][i],
            vowel_logits=out["vowel_logits"][i],
            stress_logits=out["stress_logits"][i],
        ))
    return preds


def evaluate(model, eval_loader, device, fp16: bool, tokenizer) -> dict:
    model.eval()
    tokenizer_vocab = id_to_token(tokenizer)
    total_loss = 0.0
    consonant_acc_sum = vowel_acc_sum = stress_acc_sum = 0.0
    n = 0
    refs, hyps = [], []

    with torch.no_grad():
        for batch in eval_loader:
            texts = batch.pop("texts")
            phonemes = batch.pop("phonemes")
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast("cuda", enabled=fp16):
                out = model(**batch, tokenizer_vocab=tokenizer_vocab)

            total_loss += out["loss"].item()
            consonant_acc_sum += compute_accuracy(out["consonant_logits"], batch["consonant_labels"])
            vowel_acc_sum += compute_accuracy(out["vowel_logits"], batch["vowel_labels"])
            stress_acc_sum += compute_accuracy(out["stress_logits"], batch["stress_labels"])
            n += 1

            refs.extend(phonemes)
            hyps.extend(decode_batch(texts, out, tokenizer))

    model.train()
    return {
        "eval_loss": total_loss / n,
        "consonant_acc": consonant_acc_sum / n,
        "vowel_acc": vowel_acc_sum / n,
        "stress_acc": stress_acc_sum / n,
        "mean_acc": (consonant_acc_sum + vowel_acc_sum + stress_acc_sum) / (3 * n),
        "cer": jiwer.cer(refs, hyps),
        "wer": jiwer.wer(refs, hyps),
    }
