"""Train the Hebrew G2P classifier model.

Example:
    uv run src/train.py \
        --train-dataset data/.cache/classifier-train \
        --eval-dataset data/.cache/classifier-val \
        --output-dir runs/g2p-classifier

Multi-GPU:
    accelerate launch src/train.py \
        --train-dataset data/.cache/train \
        --eval-dataset data/.cache/val \
        --output-dir runs/g2p-classifier
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from accelerate.utils import broadcast
from tqdm import tqdm

from safetensors.torch import load_file

from checkpoint import resume_step
from config import parse_args
from data import make_dataloaders
from eval import evaluate_and_save, make_eval_state
from metrics import log_train_metrics
from model import G2PModel
from optimizer import build_optimizer, build_scheduler
from tokenization import load_tokenizer


def should_eval_now(accelerator: Accelerator, last_eval_at: float, interval_seconds: float, device) -> bool:
    due = time.monotonic() - last_eval_at >= interval_seconds if accelerator.is_main_process else False
    due_tensor = torch.tensor(int(due), device=device)
    return bool(broadcast(due_tensor, from_process=0).item())


def broadcast_stop(accelerator: Accelerator, stop: bool, device) -> bool:
    stop_tensor = torch.tensor(int(stop), device=device)
    return bool(broadcast(stop_tensor, from_process=0).item())


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

    eval_state = make_eval_state()
    opt_step = 0
    if args.resume and not args.reset_steps:
        opt_step = resume_step(args.resume, scheduler)
        if accelerator.is_main_process:
            print(f"Resumed from step {opt_step}")

    global_step = opt_step * args.gradient_accumulation_steps
    optimizer.zero_grad()
    eval_interval_seconds = args.eval_seconds
    last_eval_at = time.monotonic()

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

                if should_eval_now(accelerator, last_eval_at, eval_interval_seconds, device):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        should_stop = evaluate_and_save(
                            accelerator,
                            model,
                            eval_loader,
                            device,
                            args,
                            tokenizer,
                            writer,
                            output_dir,
                            opt_step,
                            f"step {opt_step}",
                            eval_state,
                        )
                    else:
                        should_stop = False
                    accelerator.wait_for_everyone()
                    should_stop = broadcast_stop(accelerator, should_stop, device)
                    last_eval_at = time.monotonic()
                    if should_stop:
                        accelerator.wait_for_everyone()
                        if writer is not None:
                            writer.close()
                        return

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        evaluate_and_save(
            accelerator,
            model,
            eval_loader,
            device,
            args,
            tokenizer,
            writer,
            output_dir,
            opt_step,
            "final",
            eval_state,
        )
        writer.close()
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
