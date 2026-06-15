"""Evaluation helpers for the Hebrew G2P classifier."""

from __future__ import annotations

from pathlib import Path

import torch
import jiwer

from checkpoint import save_named_checkpoint
from constants import IGNORE_INDEX
from decoder import decode
from metrics import log_eval_metrics
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


def make_eval_state() -> dict:
    return {
        "best_wer": float("inf"),
        "best_cer": float("inf"),
        "best_loss": float("inf"),
        "best_acc": 0.0,
        "best_wer_step": 0,
        "no_improve_count": 0,
    }


def evaluate_and_save(
    accelerator,
    model,
    eval_loader,
    device,
    args,
    tokenizer,
    writer,
    output_dir: Path,
    opt_step: int,
    label: str,
    state: dict,
) -> None:
    metrics = evaluate(accelerator.unwrap_model(model), eval_loader, device, args.fp16, tokenizer)
    log_eval_metrics(metrics, writer, opt_step, label)
    unwrapped_model = accelerator.unwrap_model(model)
    if args.save_last:
        save_named_checkpoint(unwrapped_model, tokenizer, output_dir / "last", opt_step, metrics)
        print(f"[step {opt_step}] saved last to {output_dir}/last")
    if args.save_best_cer and metrics["cer"] < state["best_cer"]:
        state["best_cer"] = metrics["cer"]
        save_named_checkpoint(unwrapped_model, tokenizer, output_dir / "best_cer", opt_step, metrics)
        print(f"[step {opt_step}] New best CER={metrics['cer']:.4f} -> saved to {output_dir}/best_cer")
    if args.save_best_wer and metrics["wer"] < state["best_wer"]:
        save_named_checkpoint(unwrapped_model, tokenizer, output_dir / "best_wer", opt_step, metrics)
        print(f"[step {opt_step}] New best WER={metrics['wer']:.4f} -> saved to {output_dir}/best_wer")
    if args.save_best_loss and metrics["eval_loss"] < state["best_loss"]:
        state["best_loss"] = metrics["eval_loss"]
        save_named_checkpoint(unwrapped_model, tokenizer, output_dir / "best_loss", opt_step, metrics)
        print(f"[step {opt_step}] New best loss={metrics['eval_loss']:.4f} -> saved to {output_dir}/best_loss")
    if metrics["wer"] < state["best_wer"]:
        state["best_wer"] = metrics["wer"]
        state["best_acc"] = 1.0 - metrics["wer"]
        state["best_wer_step"] = opt_step
        state["no_improve_count"] = 0
        print(f"[step {opt_step}] word acc: {(1.0 - metrics['wer']) * 100:.2f}%  new best")
    else:
        state["no_improve_count"] += 1
        print(
            f"[step {opt_step}] word acc: {(1.0 - metrics['wer']) * 100:.2f}%  "
            f"best: {state['best_acc'] * 100:.2f}% @ step {state['best_wer_step']}  "
            f"(stuck for {state['no_improve_count']} evals)"
        )
