"""Checkpoint saving and resuming for classifier training."""

from __future__ import annotations

import json
from pathlib import Path


def save_named_checkpoint(model, tokenizer, checkpoint_dir: Path, step: int, metrics: dict) -> None:
    """Overwrite a stable checkpoint directory such as last, best_cer, or best_wer."""
    from safetensors.torch import save_file

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(checkpoint_dir / "model.safetensors"))
    tokenizer.save_pretrained(str(checkpoint_dir))
    payload = {"step": step, **metrics}
    (checkpoint_dir / "train_state.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def resume_step(checkpoint: str, scheduler) -> int:
    """Load step counter from a checkpoint and fast-forward the scheduler. Returns the step."""
    state_path = Path(checkpoint) / "train_state.json"
    if not state_path.exists():
        return 0
    saved = json.loads(state_path.read_text())
    step = saved["step"]
    for _ in range(step):
        scheduler.step()
    return step
