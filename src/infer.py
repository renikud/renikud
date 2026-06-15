"""Run inference with the Hebrew G2P classifier model.

Usage:
    uv run src/infer.py --checkpoint runs/g2p-classifier/step-5000 --text "שלום עולם"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from constants import MAX_LEN
from decoder import decode
from model import G2PModel
from phonology import normalize_graphemes
from tokenization import load_tokenizer, id_to_token


def parse_args():
    parser = argparse.ArgumentParser(description="Infer IPA from Hebrew text using classifier model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    return parser.parse_args()


def load_checkpoint(model: G2PModel, checkpoint_dir: str) -> None:
    state = load_file(str(Path(checkpoint_dir) / "model.safetensors"), device="cpu")
    model.load_state_dict(state)


def phonemize(text: str, model: G2PModel, tokenizer, device: torch.device, max_len: int) -> str:
    """Convert unvocalized Hebrew text to IPA using the classifier model."""
    text = normalize_graphemes(text)

    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()  # [S, 2]
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer_vocab=id_to_token(tokenizer),
        )

    return decode(
        text=text,
        offset_mapping=offset_mapping,
        consonant_logits=out["consonant_logits"][0],
        vowel_logits=out["vowel_logits"][0],
        stress_logits=out["stress_logits"][0],
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer()
    model = G2PModel()
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    print(phonemize(args.text, model, tokenizer, device, args.max_len))


if __name__ == "__main__":
    main()
