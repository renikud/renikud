---
name: vast-ai-train
description: Prepare and run model training on a Vast.ai remote GPU machine through a tmux SSH pane. Use when the user asks to connect to a Vast.ai host, clone the current repo/branch under /workspace, install uv and dataset extraction tools, authenticate to Hugging Face with a user-provided token, download/extract training data, prepare eval data, and launch remote pretraining or training commands.
---

# Vast AI Train

## Core Rules

- Use an existing tmux pane when the user asks for one; otherwise create a new pane/window for the SSH session.
- Connect with the username, host, port, and SSH key the user provides. Vast keys usually live under `~/.ssh`; inspect filenames without printing key material.
- After connecting, send all remote commands through the tmux pane unless the user explicitly asks for direct `ssh` commands.
- On every new Vast instance, read `/etc/vast-agents-guide.md` before acting beyond connection/setup.
- Keep all repo files and datasets under `/workspace` on the remote.
- Do not hardcode secrets in skill files, scripts, repo files, or command files. If a Hugging Face token is supplied in the prompt, use it only at runtime and avoid echoing it into shell history.

## Connect

If the pane is already connected, use it. Otherwise send an SSH command into the selected pane:

```bash
ssh -i ~/.ssh/<vast-key> -p <port> <user>@<host> -L 8080:localhost:8080
```

Once connected:

```bash
cat /etc/vast-agents-guide.md
```

Read enough of the output to confirm the image rules, Python environment, storage notes, and CUDA/PyTorch guidance.

## Base Setup

Run setup from the remote tmux pane. Install `p7zip-full` before dataset extraction and install `uv` after removing the default venv:

```bash
cd /workspace
apt-get update && apt-get install -y p7zip-full wget git
rm -rf /venv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc 2>/dev/null || true
source "$HOME/.local/bin/env" 2>/dev/null || true
cd /workspace
```

## Clone Current Repo Branch

Determine the local repo URL and branch before sending the remote command:

```bash
git remote get-url origin
git branch --show-current
```

Then clone or refresh the remote checkout under `/workspace/<repo-name>`:

```bash
cd /workspace
if [ -d <repo-name>/.git ]; then
  cd <repo-name>
  git fetch origin <branch>
  git switch <branch> || git switch -c <branch> --track origin/<branch>
  git reset --hard origin/<branch>
else
  git clone --branch <branch> <repo-url> <repo-name>
  cd <repo-name>
fi
git status -sb
uv sync
```

## Hugging Face Login

Use a token provided by the user at runtime. Never place the token in `SKILL.md`, committed files, or shell history.

Preferred tmux-safe flow:

```bash
read -rsp "HF token: " HF_TOKEN; echo
export HF_TOKEN
uv run hf auth login --token "$HF_TOKEN"
```

If the user has already configured HF auth on the instance, verify without exposing the token:

```bash
uv run hf auth whoami
```

## Download Dataset And Eval

For a Hugging Face dataset repo such as `thewh1teagle/renikud-data`, download into `data/`:

```bash
cd /workspace/<repo-name>
mkdir -p data
uv run hf download <dataset-repo> --repo-type dataset --local-dir data
```

Extract `.7z` datasets in `data/`:

```bash
cd /workspace/<repo-name>/data
for f in *.7z; do [ -e "$f" ] && 7z x -y "$f"; done
```

Download the G2P benchmark eval file:

```bash
cd /workspace/<repo-name>/data
wget -O gt.tsv https://github.com/phonikud/heb-g2p-benchmark/raw/refs/heads/main/web/data/gt.tsv
```

## Prepare And Train

For the Knesset Vox pretrain task, use `data/knesset_vox_extracted.tsv` as train and `data/gt.tsv` as eval. Prepare aligned JSONL and token caches with the repo scripts:

```bash
cd /workspace/<repo-name>
WORKERS=$(nproc)
uv run scripts/prepare_align.py data/knesset_vox_extracted.tsv data/knesset_vox_alignment.jsonl --workers "$WORKERS" --chunk_size 1000
uv run scripts/prepare_align.py data/gt.tsv data/gt_alignment.jsonl --workers 2 --chunk_size 100
uv run scripts/prepare_tokens.py data/knesset_vox_alignment.jsonl data/.cache/knesset_vox_train_menaked --workers "$WORKERS" --batch_size 1000
uv run scripts/prepare_tokens.py data/gt_alignment.jsonl data/.cache/gt_eval_menaked --workers 2 --batch_size 250
```

Start training with timed eval every five minutes. Set batch size from the user request or GPU memory; use `128` when the user asks for a larger batch and memory allows it:

```bash
uv run accelerate launch src/train.py \
  --train-dataset data/.cache/knesset_vox_train_menaked \
  --eval-dataset data/.cache/gt_eval_menaked \
  --output-dir runs/g2p-menaked-knesset-vox-pretrain \
  --lr 1e-4 \
  --train-batch-size 128 \
  --eval-batch-size 128 \
  --eval-seconds 300
```

If the user asks for a single line, collapse the command without changing arguments.

## Monitoring

Use tmux pane capture for status:

```bash
tmux capture-pane -t <pane> -p -S -80
```

Watch for:

- alignment counts and failures
- token cache save completion
- `Trainable params` / `frozen encoder params`
- timed eval metrics: loss, CER, WER, word accuracy
- CUDA OOM; if it happens, reduce `--train-batch-size` and use a new output dir

If the SSH pane is closed, reconnect with the same key/host/port and inspect the remote tmux session before starting duplicate work.
