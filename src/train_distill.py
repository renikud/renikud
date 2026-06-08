"""Distill the DictaBERT-large-char G2P teacher into a small ModernBERT student.

Example:
    uv run accelerate launch src/train_distill.py \
        --train-dataset dataset/.cache/train \
        --eval-dataset dataset/.cache/val \
        --teacher-checkpoint models/renikud-dictabert-large-char \
        --output-dir outputs/g2p-modernbert-distill
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import jiwer
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from checkpoint import resume_step, save_named_checkpoint
from constants import IGNORE_INDEX, MAX_LEN
from data import make_dataloaders
from decoder import decode
from eval import evaluate
from metrics import log_eval_metrics, log_train_metrics
from model import G2PModel
from modernbert_encoder import build_encoder as build_modernbert_encoder
from optimizer import build_optimizer, build_scheduler
from phonology import (
    NUM_CONSONANT_CLASSES,
    NUM_STRESS_CLASSES,
    NUM_VOWEL_CLASSES,
    apply_consonant_mask,
    build_consonant_mask,
    is_hebrew_letter,
    normalize_graphemes,
)
from tokenization import id_to_token, load_tokenizer


TEACHER_LAYER_COUNT = 24
DEFAULT_GT_URL = "https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/web/data/gt.tsv"


class StudentG2PModel(nn.Module):
    """ModernBERT student with the same coupled G2P heads as the teacher."""

    def __init__(self, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.encoder = build_modernbert_encoder()
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.consonant_head = nn.Linear(hidden_size, NUM_CONSONANT_CLASSES)
        self.vowel_head = nn.Linear(hidden_size + NUM_CONSONANT_CLASSES, NUM_VOWEL_CLASSES)
        self.stress_head = nn.Linear(hidden_size + NUM_CONSONANT_CLASSES + NUM_VOWEL_CLASSES, NUM_STRESS_CLASSES)
        self.register_buffer("_consonant_mask", build_consonant_mask())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        consonant_labels: torch.Tensor | None = None,
        vowel_labels: torch.Tensor | None = None,
        stress_labels: torch.Tensor | None = None,
        tokenizer_vocab: dict[int, str] | None = None,
        output_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=output_hidden_states,
        )
        hidden = self.dropout(encoder_outputs.last_hidden_state)
        consonant_logits = self.consonant_head(hidden)
        vowel_logits = self.vowel_head(torch.cat([hidden, consonant_logits], dim=-1))
        stress_logits = self.stress_head(torch.cat([hidden, consonant_logits, vowel_logits], dim=-1))

        if tokenizer_vocab is not None:
            consonant_logits = apply_consonant_mask(consonant_logits, input_ids, tokenizer_vocab, self._consonant_mask)

        output: dict[str, torch.Tensor | tuple[torch.Tensor, ...]] = {
            "consonant_logits": consonant_logits,
            "vowel_logits": vowel_logits,
            "stress_logits": stress_logits,
        }
        if output_hidden_states:
            output["hidden_states"] = encoder_outputs.hidden_states

        if consonant_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            output["loss"] = (
                loss_fct(consonant_logits.view(-1, NUM_CONSONANT_CLASSES), consonant_labels.view(-1))
                + loss_fct(vowel_logits.view(-1, NUM_VOWEL_CLASSES), vowel_labels.view(-1))
                + loss_fct(stress_logits.view(-1, NUM_STRESS_CLASSES), stress_labels.view(-1))
            )

        return output


class HiddenProjectors(nn.Module):
    """Train-only adapters that project selected teacher layers into student width."""

    def __init__(self, teacher_hidden_size: int, student_hidden_size: int, student_layers: int) -> None:
        super().__init__()
        self.teacher_layer_ids = teacher_layer_ids(student_layers)
        self.projectors = nn.ModuleList(
            nn.Linear(teacher_hidden_size, student_hidden_size) for _ in self.teacher_layer_ids
        )

    def forward(
        self,
        teacher_hidden_states: tuple[torch.Tensor, ...],
        student_hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.bool()
        losses = []
        for student_idx, (teacher_idx, projector) in enumerate(zip(self.teacher_layer_ids, self.projectors), start=1):
            teacher_hidden = projector(teacher_hidden_states[teacher_idx])
            student_hidden = student_hidden_states[student_idx]
            cosine = F.cosine_similarity(student_hidden, teacher_hidden, dim=-1)
            losses.append((1.0 - cosine)[mask].mean())
        return torch.stack(losses).mean()


def teacher_layer_ids(student_layers: int) -> list[int]:
    return [
        round((idx + 1) * TEACHER_LAYER_COUNT / student_layers)
        for idx in range(student_layers)
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Distill the Hebrew G2P classifier model")
    parser.add_argument("--train-dataset", type=str, default=None)
    parser.add_argument("--eval-dataset", type=str, default=None)
    parser.add_argument("--train-tsv", type=str, default=None, help="Raw Hebrew<TAB>phonemes TSV for pure distillation")
    parser.add_argument("--benchmark-gt", type=str, default="gt.tsv", help="External Hebrew<TAB>phonemes benchmark TSV")
    parser.add_argument("--teacher-checkpoint", type=str, default="models/renikud-dictabert-large-char")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--encoder-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--projector-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--consonant-kl-weight", type=float, default=1.0)
    parser.add_argument("--vowel-kl-weight", type=float, default=2.0)
    parser.add_argument("--stress-kl-weight", type=float, default=1.5)
    parser.add_argument("--ce-weight", type=float, default=0.25)
    parser.add_argument("--hidden-weight", type=float, default=0.5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reset-steps", action="store_true", default=False)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=torch.cuda.is_available())
    parser.add_argument("--save-last", action="store_true", default=False)
    parser.add_argument("--save-best-cer", action="store_true", default=False)
    parser.add_argument("--save-best-wer", action="store_true", default=False)
    parser.add_argument("--save-best-loss", action="store_true", default=False)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    return parser.parse_args()


class RawTsvDistillDataset(IterableDataset):
    """Stream raw Hebrew sentences from Hebrew<TAB>phonemes TSV."""

    def __init__(self, path: str) -> None:
        self.path = path

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1

        with open(self.path, encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle):
                if line_idx % num_workers != worker_id:
                    continue
                line = line.rstrip("\n")
                if not line:
                    continue
                hebrew, _, phonemes = line.partition("\t")
                if hebrew:
                    yield {"hebrew": normalize_graphemes(hebrew), "phonemes": phonemes}


class RawDistillCollator:
    def __init__(self, tokenizer, max_len: int = MAX_LEN) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, features: list[dict]) -> dict:
        texts = [feature["hebrew"] for feature in features]
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_len,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encodings.pop("offset_mapping")
        distill_mask = torch.zeros_like(encodings["attention_mask"], dtype=torch.bool)
        for batch_idx, text in enumerate(texts):
            for token_idx, (start, end) in enumerate(offsets[batch_idx].tolist()):
                if end - start == 1 and start < len(text) and is_hebrew_letter(text[start]):
                    distill_mask[batch_idx, token_idx] = True

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "distill_mask": distill_mask,
        }


def count_lines(path: str) -> int:
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


def make_raw_train_loader(args, tokenizer) -> tuple[DataLoader, int]:
    line_count = count_lines(args.train_tsv)
    loader = DataLoader(
        RawTsvDistillDataset(args.train_tsv),
        batch_size=args.train_batch_size,
        collate_fn=RawDistillCollator(tokenizer),
        num_workers=args.dataloader_workers,
    )
    return loader, line_count


def load_benchmark_gt(path: str) -> list[dict[str, str]]:
    data = []
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t", fieldnames=["sentence", "phonemes"]):
            data.append({"sentence": row["sentence"], "phonemes": row["phonemes"]})
    return data


def benchmark_model(model, tokenizer, gt_data: list[dict[str, str]], device, fp16: bool) -> dict[str, float]:
    model.eval()
    refs, hyps = [], []
    tokenizer_vocab = id_to_token(tokenizer)
    with torch.no_grad():
        for item in gt_data:
            text = normalize_graphemes(item["sentence"])
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=MAX_LEN,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offset_mapping = encoding.pop("offset_mapping")[0].tolist()
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            with torch.autocast("cuda", enabled=fp16):
                out = model(input_ids=input_ids, attention_mask=attention_mask, tokenizer_vocab=tokenizer_vocab)
            hyps.append(decode(
                text=text,
                offset_mapping=offset_mapping,
                consonant_logits=out["consonant_logits"][0],
                vowel_logits=out["vowel_logits"][0],
                stress_logits=out["stress_logits"][0],
            ))
            refs.append(item["phonemes"])
    model.train()
    cer = jiwer.cer(refs, hyps)
    wer = jiwer.wer(refs, hyps)
    return {
        "eval_loss": float("nan"),
        "consonant_acc": float("nan"),
        "vowel_acc": float("nan"),
        "stress_acc": float("nan"),
        "mean_acc": float("nan"),
        "cer": cer,
        "wer": wer,
        "acc": 1.0 - wer,
    }


def kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float, mask: torch.Tensor) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    token_loss = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return token_loss[mask].mean() * temperature * temperature


def g2p_forward_with_hidden(model, batch: dict[str, torch.Tensor]) -> dict:
    encoder_outputs = model.encoder(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        return_dict=True,
        output_hidden_states=True,
    )
    hidden = model.dropout(encoder_outputs.last_hidden_state)
    consonant_logits = model.consonant_head(hidden)
    vowel_logits = model.vowel_head(torch.cat([hidden, consonant_logits], dim=-1))
    stress_logits = model.stress_head(torch.cat([hidden, consonant_logits, vowel_logits], dim=-1))
    return {
        "consonant_logits": consonant_logits,
        "vowel_logits": vowel_logits,
        "stress_logits": stress_logits,
        "hidden_states": encoder_outputs.hidden_states,
    }


def build_teacher(checkpoint_dir: str) -> G2PModel:
    teacher = G2PModel()
    state = load_file(str(Path(checkpoint_dir) / "model.safetensors"), device="cpu")
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def save_distill_checkpoint(student, projectors, tokenizer, checkpoint_dir: Path, step: int, metrics: dict) -> None:
    save_named_checkpoint(student, tokenizer, checkpoint_dir, step, metrics)
    save_file(projectors.state_dict(), str(checkpoint_dir / "projectors.safetensors"))


def load_projectors_if_present(projectors, checkpoint_dir: str) -> None:
    projectors_path = Path(checkpoint_dir) / "projectors.safetensors"
    if projectors_path.exists():
        projectors.load_state_dict(load_file(str(projectors_path), device="cpu"), strict=True)


def main():
    args = parse_args()
    if args.train_tsv is None and (args.train_dataset is None or args.eval_dataset is None):
        raise ValueError("Provide either --train-tsv for raw distillation or both --train-dataset and --eval-dataset.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")
    device = accelerator.device
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if accelerator.is_main_process else None

    tokenizer = load_tokenizer()
    raw_distill = args.train_tsv is not None
    if raw_distill:
        train_loader, train_examples = make_raw_train_loader(args, tokenizer)
        eval_loader = None
        benchmark_data = load_benchmark_gt(args.benchmark_gt) if Path(args.benchmark_gt).exists() else []
        if accelerator.is_main_process:
            print(f"Loaded raw distillation TSV: {args.train_tsv} ({train_examples:,} examples)")
            print(f"Loaded benchmark GT: {args.benchmark_gt} ({len(benchmark_data):,} examples)")
    else:
        train_loader, eval_loader = make_dataloaders(args)
        train_examples = len(train_loader.dataset)
        benchmark_data = []

    teacher = build_teacher(args.teacher_checkpoint).to(device)
    student = StudentG2PModel()
    projectors = HiddenProjectors(
        teacher_hidden_size=teacher.encoder.config.hidden_size,
        student_hidden_size=student.encoder.config.hidden_size,
        student_layers=student.encoder.config.num_hidden_layers,
    )

    if args.resume:
        state = load_file(str(Path(args.resume) / "model.safetensors"), device="cpu")
        student.load_state_dict(state, strict=False)
        load_projectors_if_present(projectors, args.resume)
        if accelerator.is_main_process:
            print(f"Loaded student weights from {args.resume}")

    steps_per_epoch = math.ceil(train_examples / args.train_batch_size)
    total_opt_steps = math.ceil(steps_per_epoch * args.epochs / args.gradient_accumulation_steps)
    optimizer = build_optimizer(student, args.encoder_lr, args.head_lr, args.weight_decay)
    optimizer.add_param_group({"params": projectors.parameters(), "lr": args.projector_lr, "weight_decay": args.weight_decay})
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_opt_steps)

    if eval_loader is None:
        student, projectors, optimizer, train_loader, scheduler = accelerator.prepare(
            student, projectors, optimizer, train_loader, scheduler
        )
    else:
        student, projectors, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
            student, projectors, optimizer, train_loader, eval_loader, scheduler
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
        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}",
            total=steps_per_epoch,
            dynamic_ncols=True,
            disable=not accelerator.is_main_process,
        )

        for batch in pbar:
            if opt_step >= total_opt_steps:
                break

            batch.pop("texts", None)
            batch.pop("phonemes", None)
            distill_mask = batch.pop("distill_mask", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            if distill_mask is not None:
                label_mask = distill_mask.to(device)
            else:
                label_mask = batch["consonant_labels"] != IGNORE_INDEX

            with torch.no_grad(), accelerator.autocast():
                teacher_out = g2p_forward_with_hidden(teacher, batch)

            with accelerator.autocast():
                student_out = student(**batch, output_hidden_states=True)
                consonant_kl = kl_loss(student_out["consonant_logits"], teacher_out["consonant_logits"], args.temperature, label_mask)
                vowel_kl = kl_loss(student_out["vowel_logits"], teacher_out["vowel_logits"], args.temperature, label_mask)
                stress_kl = kl_loss(student_out["stress_logits"], teacher_out["stress_logits"], args.temperature, label_mask)
                hidden_loss = projectors(teacher_out["hidden_states"], student_out["hidden_states"], batch["attention_mask"])
                ce_loss = student_out.get("loss")
                loss = (
                    args.consonant_kl_weight * consonant_kl
                    + args.vowel_kl_weight * vowel_kl
                    + args.stress_kl_weight * stress_kl
                    + args.hidden_weight * hidden_loss
                )
                if ce_loss is not None and args.ce_weight > 0:
                    loss = loss + args.ce_weight * ce_loss

            scaled_loss = loss / args.gradient_accumulation_steps
            accelerator.backward(scaled_loss)
            epoch_loss_sum += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % args.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(
                    list(student.parameters()) + list(projectors.parameters()),
                    args.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                train_loss = epoch_loss_sum / epoch_steps
                pbar.set_postfix(
                    step=opt_step,
                    loss=f"{train_loss:.4f}",
                    kl=f"{(consonant_kl + vowel_kl + stress_kl).item():.4f}",
                    hid=f"{hidden_loss.item():.4f}",
                    enc_lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    head_lr=f"{optimizer.param_groups[2]['lr']:.2e}",
                )

                if accelerator.is_main_process:
                    if opt_step % args.logging_steps == 0:
                        log_train_metrics(train_loss, optimizer.param_groups[0]["lr"], optimizer.param_groups[2]["lr"], writer, opt_step)
                        writer.add_scalar("train/distill_consonant_kl", consonant_kl.item(), opt_step)
                        writer.add_scalar("train/distill_vowel_kl", vowel_kl.item(), opt_step)
                        writer.add_scalar("train/distill_stress_kl", stress_kl.item(), opt_step)
                        writer.add_scalar("train/distill_hidden", hidden_loss.item(), opt_step)

                    if opt_step % args.save_steps == 0:
                        unwrapped_student = accelerator.unwrap_model(student)
                        if raw_distill:
                            metrics = benchmark_model(unwrapped_student, tokenizer, benchmark_data, device, args.fp16)
                        else:
                            metrics = evaluate(unwrapped_student, eval_loader, device, args.fp16, tokenizer)
                        log_eval_metrics(metrics, writer, opt_step, f"step {opt_step}")
                        if args.save_last:
                            save_distill_checkpoint(unwrapped_student, accelerator.unwrap_model(projectors), tokenizer, output_dir / "last", opt_step, metrics)
                            print(f"[step {opt_step}] saved last to {output_dir}/last")
                        if args.save_best_cer and metrics["cer"] < best_cer:
                            best_cer = metrics["cer"]
                            save_distill_checkpoint(unwrapped_student, accelerator.unwrap_model(projectors), tokenizer, output_dir / "best_cer", opt_step, metrics)
                            print(f"[step {opt_step}] New best CER={metrics['cer']:.4f} -> saved to {output_dir}/best_cer")
                        if args.save_best_wer and metrics["wer"] < best_wer:
                            save_distill_checkpoint(unwrapped_student, accelerator.unwrap_model(projectors), tokenizer, output_dir / "best_wer", opt_step, metrics)
                            print(f"[step {opt_step}] New best WER={metrics['wer']:.4f} -> saved to {output_dir}/best_wer")
                        if args.save_best_loss and metrics["eval_loss"] < best_loss:
                            best_loss = metrics["eval_loss"]
                            save_distill_checkpoint(unwrapped_student, accelerator.unwrap_model(projectors), tokenizer, output_dir / "best_loss", opt_step, metrics)
                            print(f"[step {opt_step}] New best loss={metrics['eval_loss']:.4f} -> saved to {output_dir}/best_loss")
                        if metrics["wer"] < best_wer:
                            best_wer = metrics["wer"]
                            best_acc = 1.0 - metrics["wer"]
                            best_wer_step = opt_step
                            no_improve_count = 0
                            print(f"[step {opt_step}] word acc: {(1.0 - metrics['wer']) * 100:.2f}%  new best")
                        else:
                            no_improve_count += 1
                            print(f"[step {opt_step}] word acc: {(1.0 - metrics['wer']) * 100:.2f}%  best: {best_acc * 100:.2f}% @ step {best_wer_step}  (stuck for {no_improve_count} evals)")

    if accelerator.is_main_process:
        if raw_distill:
            metrics = benchmark_model(accelerator.unwrap_model(student), tokenizer, benchmark_data, device, args.fp16)
        else:
            metrics = evaluate(accelerator.unwrap_model(student), eval_loader, device, args.fp16, tokenizer)
        log_eval_metrics(metrics, writer, opt_step, "final")
        if args.save_last:
            save_distill_checkpoint(accelerator.unwrap_model(student), accelerator.unwrap_model(projectors), tokenizer, output_dir / "last", opt_step, metrics)
        writer.close()


if __name__ == "__main__":
    main()
