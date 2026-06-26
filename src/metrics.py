"""Metric logging helpers for training."""

from __future__ import annotations


def log_train_metrics(loss, lr, writer, opt_step):
    print(f"[step {opt_step}] train_loss={loss:.4f} lr={lr:.2e}")
    writer.add_scalar("train/loss", loss, opt_step)
    writer.add_scalar("train/lr", lr, opt_step)


def log_eval_metrics(metrics, writer, opt_step, label):
    print(f"[{label}] loss={metrics['eval_loss']:.4f} consonant={metrics['consonant_acc']:.1%} vowel={metrics['vowel_acc']:.1%} stress={metrics['stress_acc']:.1%} char_acc={1-metrics['cer']:.1%} word_acc={1-metrics['wer']:.1%}")
    if writer is not None:
        writer.add_scalar("eval/loss", metrics["eval_loss"], opt_step)
        writer.add_scalar("eval/consonant_acc", metrics["consonant_acc"], opt_step)
        writer.add_scalar("eval/vowel_acc", metrics["vowel_acc"], opt_step)
        writer.add_scalar("eval/stress_acc", metrics["stress_acc"], opt_step)
        writer.add_scalar("eval/mean_acc", metrics["mean_acc"], opt_step)
        writer.add_scalar("eval/char_acc", 1 - metrics["cer"], opt_step)
        writer.add_scalar("eval/word_acc", 1 - metrics["wer"], opt_step)
