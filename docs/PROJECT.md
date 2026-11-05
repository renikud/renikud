# Project rules

Rules for agents on this project. General workflow rules live in [`../AGENTS.md`](../AGENTS.md).

See [ARCHITECTURE.md](ARCHITECTURE.md) for what the model is.

## ML model development

- `accelerate` for DDP from the start, not retrofitted.
- Time-based eval, driven by accelerate.
- Eval logged to TensorBoard through accelerate.
- TensorBoard logs and checkpoints go to `runs/`. Every eval writes `last`; `best` is updated only when the eval loss improves, never on a training-loss or wall-clock basis.
- Three modules: model, data, training loop. Nothing else, unless really needed.
- `src/` holds only what every run needs, never one-off branches. Steps needed to reproduce the model from scratch (tokenizer, init checkpoint, data download, export, quantization) are standalone scripts in `scripts/`; scratch work, probes and one-off fixups (checkpoint surgery, optimizer regrouping) stay in `plans/<name>/`.
- Data loader stays general purpose — one simple format, never shaped to a specific dataset.
- Preprocess on the fly so training starts immediately.
- Fast epochs from the start (bucketing and the like), but no slow-compile startup cost.
- `tqdm` per epoch with live loss, and per eval.
- Wire up [pyinject](https://github.com/thewh1teagle/pyinject) in the training loop — `listen()` at startup, `poll()` in the step loop — so a live run stays inspectable.
- Never lose steps. Time is money: a run is stopped only after a checkpoint, and it comes back with `--resume`, never from scratch. Fixes to the loop are applied through a resume.
- Data order is deterministic: the loader, sampler, split and masking are seeded, so a run's training and eval rows are reproducible from its seed and step count, never a leak into eval.
- Heads and adapters are sized by measurement, not by feel — neither starved nor bloated. Known configurations usually have a golden size; start from it and only move on a measured gain.

## Commands

- No long sleeps — background the command, the user is watching the screen.
- Anything over ~20 s runs in tmux or in the background with `tee` to a log under `runs/`, never through a buffering pipe such as `| tail`. Progress must be readable from the log tail at any moment, and the tail is reported, not waited on.
- Prefer the fastest check that answers the same question with the same certainty: a 40-row slice over the full file, one step over an epoch, a cached tensor over a recompute. The user is waiting on every command.

## Sizing

- No width, depth, head count, embedding size or hidden size is decided by feel. Every size is a config entry and a `--flag` on the trainer, and the default is what `scripts/sweep_capacity.py` measured: equal wall-clock budget per candidate, eval loss against parameter count, pick the knee. A size that was never swept is a placeholder and says so in a comment.
- Sweep again when the data, the objective or a neighbouring module changes; a size measured for one setup is a guess for the next.
- Record each sweep's table under `plans/` with the date, the budget and the pick, so the next person can see why the default is what it is.
