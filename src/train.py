"""Train the Hebrew G2P classifier model.

Example:
    uv run src/train.py \
        --train-dataset dataset/.cache/classifier-train \
        --eval-dataset dataset/.cache/classifier-val \
        --output-dir outputs/g2p-classifier

Multi-GPU:
    accelerate launch src/train.py \
        --train-dataset dataset/.cache/train \
        --eval-dataset dataset/.cache/val \
        --output-dir outputs/g2p-classifier
"""

from __future__ import annotations

import math
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from tqdm import tqdm

from safetensors.torch import load_file

from checkpoint import resume_step, save_named_checkpoint
from config import parse_args
from data import make_dataloaders
from eval import evaluate
from metrics import log_train_metrics, log_eval_metrics
from model import G2PModel
from optimizer import build_optimizer, build_scheduler
from tokenization import load_tokenizer


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")
    device = accelerator.device

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if accelerator.is_main_process else None

    tokenizer = load_tokenizer()

    train_loader, eval_loader = make_dataloaders(args)

    model = G2PModel(flash_attention=args.flash_attention)

    if args.resume:
        state = load_file(str(Path(args.resume) / "model.safetensors"), device="cpu")
        model.load_state_dict(state, strict=False)
        if accelerator.is_main_process:
            print(f"Loaded weights from {args.resume}")

    if args.freeze_encoder_steps > 0:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        if accelerator.is_main_process:
            print("Encoder frozen.")

    total_opt_steps = math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation_steps)
    optimizer = build_optimizer(model, args.encoder_lr, args.head_lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_opt_steps)

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    best_wer = float("inf")
    best_cer = float("inf")
    best_loss = float("inf")
    best_acc = 0.0
    best_wer_step = 0
    no_improve_count = 0
    opt_step = 0
    if args.resume and not args.reset_steps:
        opt_step = resume_step(args.resume, scheduler)
        if accelerator.is_main_process:
            print(f"Resumed from step {opt_step}")

    global_step = opt_step * args.gradient_accumulation_steps
    optimizer.zero_grad()

    for epoch in range(math.ceil(args.epochs)):
        epoch_loss_sum = 0.0
        epoch_steps = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}", dynamic_ncols=True, disable=not accelerator.is_main_process)

        for batch in pbar:
            if opt_step >= total_opt_steps:
                break

            if args.freeze_encoder_steps > 0 and global_step == args.freeze_encoder_steps:
                for p in accelerator.unwrap_model(model).encoder.parameters():
                    p.requires_grad_(True)
                if accelerator.is_main_process:
                    print(f"\n[step {opt_step}] Encoder unfrozen.")

            batch.pop("texts")
            batch.pop("phonemes")
            with accelerator.autocast():
                out = model(**batch)

            scaled_loss = out["loss"] / args.gradient_accumulation_steps
            accelerator.backward(scaled_loss)
            epoch_loss_sum += out["loss"].item()
            epoch_steps += 1
            global_step += 1

            if global_step % args.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                train_loss = epoch_loss_sum / epoch_steps
                pbar.set_postfix(
                    step=opt_step,
                    loss=f"{train_loss:.4f}",
                    enc_lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    head_lr=f"{optimizer.param_groups[2]['lr']:.2e}",
                )

                if accelerator.is_main_process:
                    if opt_step % args.logging_steps == 0:
                        log_train_metrics(train_loss, optimizer.param_groups[0]["lr"], optimizer.param_groups[2]["lr"], writer, opt_step)

                    if opt_step % args.save_steps == 0:
                        metrics = evaluate(accelerator.unwrap_model(model), eval_loader, device, args.fp16, tokenizer)
                        log_eval_metrics(metrics, writer, opt_step, f"step {opt_step}")
                        unwrapped_model = accelerator.unwrap_model(model)
                        if args.save_last:
                            save_named_checkpoint(unwrapped_model, tokenizer, output_dir / "last", opt_step, metrics)
                            print(f"[step {opt_step}] saved last to {output_dir}/last")
                        if args.save_best_cer and metrics["cer"] < best_cer:
                            best_cer = metrics["cer"]
                            save_named_checkpoint(unwrapped_model, tokenizer, output_dir / "best_cer", opt_step, metrics)
                            print(f"[step {opt_step}] New best CER={metrics['cer']:.4f} → saved to {output_dir}/best_cer")
                        if args.save_best_wer and metrics["wer"] < best_wer:
                            save_named_checkpoint(unwrapped_model, tokenizer, output_dir / "best_wer", opt_step, metrics)
                            print(f"[step {opt_step}] New best WER={metrics['wer']:.4f} → saved to {output_dir}/best_wer")
                        if args.save_best_loss and metrics["eval_loss"] < best_loss:
                            best_loss = metrics["eval_loss"]
                            save_named_checkpoint(unwrapped_model, tokenizer, output_dir / "best_loss", opt_step, metrics)
                            print(f"[step {opt_step}] New best loss={metrics['eval_loss']:.4f} → saved to {output_dir}/best_loss")
                        if metrics["wer"] < best_wer:
                            best_wer = metrics["wer"]
                            best_acc = 1.0 - metrics["wer"]
                            best_wer_step = opt_step
                            no_improve_count = 0
                            print(f"[step {opt_step}] word acc: {(1.0 - metrics['wer']) * 100:.2f}%  ↑ new best!")
                        else:
                            no_improve_count += 1
                            print(f"[step {opt_step}] word acc: {(1.0 - metrics['wer']) * 100:.2f}%  ↓ best: {best_acc * 100:.2f}% @ step {best_wer_step}  (stuck for {no_improve_count} evals)")

    if accelerator.is_main_process:
        metrics = evaluate(accelerator.unwrap_model(model), eval_loader, device, args.fp16, tokenizer)
        log_eval_metrics(metrics, writer, opt_step, "final")
        if args.save_last:
            save_named_checkpoint(accelerator.unwrap_model(model), tokenizer, output_dir / "last", opt_step, metrics)
        writer.close()


if __name__ == "__main__":
    main()
