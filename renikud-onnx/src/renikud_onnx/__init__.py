"""renikud-onnx: Hebrew grapheme-to-phoneme inference via ONNX."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort

ALEF_ORD = ord("א")
TAF_ORD = ord("ת")
STRESS_MARK = "ˈ"
ORTHOGRAPHIC_MARKERS = ("'", '"')
HF_REPO_ID = "renikud/renikud"
HF_MODEL_FILENAME = "model.onnx"


def _is_hebrew(char: str) -> bool:
    return ALEF_ORD <= ord(char) <= TAF_ORD


def normalize_graphemes(text: str) -> str:
    text = re.sub(r"[׳'`´]", "'", text)
    text = re.sub(r'[״""]', '"', text)
    return text


def download_model(repo_id: str = HF_REPO_ID, filename: str = HF_MODEL_FILENAME) -> str:
    """Download the ONNX model from the Hugging Face Hub (cached after the first call)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


class G2P:
    def __init__(self, model_path: str | Path | None = None) -> None:
        """Load the G2P model.

        Args:
            model_path: Path to a local ``model.onnx``. If omitted, the model is
                fetched from the Hugging Face Hub and cached locally.
        """
        if model_path is None:
            model_path = download_model()
        self._session = ort.InferenceSession(str(model_path))
        meta = self._session.get_modelmeta().custom_metadata_map
        self._vocab: dict[str, int] = json.loads(meta["vocab"])
        self._consonant_vocab: dict[int, str] = {int(k): v for k, v in json.loads(meta["consonant_vocab"]).items()}
        self._vowel_vocab: dict[int, str] = {int(k): v for k, v in json.loads(meta["vowel_vocab"]).items()}
        self._cls_id = int(meta["cls_token_id"])
        self._sep_id = int(meta["sep_token_id"])
        self._letter_constraints: dict[str, list[int]] = {
            k: v for k, v in json.loads(meta["letter_consonant_constraints"]).items()
        }
        self._geresh_map: dict[str, str] = json.loads(meta.get("geresh_map", "{}"))

    def _tokenize(self, text: str) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        """Tokenize character by character, return ids, mask, and offset mapping."""
        normalized = unicodedata.normalize("NFD", text)
        unk_id = self._vocab.get("[UNK]", 0)
        ids = [self._cls_id]
        offsets = [(0, 0)]  # CLS
        for i, c in enumerate(normalized):
            ids.append(self._vocab.get(c, unk_id))
            offsets.append((i, i + 1))
        ids.append(self._sep_id)
        offsets.append((0, 0))  # SEP
        mask = [1] * len(ids)
        return ids, mask, offsets

    def _best_stress_per_word(self, offsets: list[tuple[int, int]], text: str, stress_logits: np.ndarray) -> set[int]:
        word_spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        words: dict[int, list[int]] = {i: [] for i in range(len(word_spans))}
        for tok_idx, (start, end) in enumerate(offsets):
            if end - start != 1:
                continue
            for word_idx, (ws, we) in enumerate(word_spans):
                if ws <= start < we:
                    words[word_idx].append(tok_idx)
                    break
        stressed: set[int] = set()
        for toks in words.values():
            if toks:
                stressed.add(max(toks, key=lambda t: stress_logits[t, 1]))
        return stressed

    def phonemize(self, text: str) -> str:
        text = normalize_graphemes(text)
        normalized = unicodedata.normalize("NFD", text)
        ids, mask, offsets = self._tokenize(text)

        consonant_logits, vowel_logits, stress_logits = self._session.run(
            ["consonant_logits", "vowel_logits", "stress_logits"],
            {
                "input_ids": np.array([ids], dtype=np.int64),
                "attention_mask": np.array([mask], dtype=np.int64),
            },
        )
        # logits shape: [1, seq_len, num_classes]
        consonant_preds = consonant_logits[0].argmax(axis=-1)
        vowel_preds = vowel_logits[0].argmax(axis=-1)
        stressed_positions = self._best_stress_per_word(offsets, normalized, stress_logits[0])

        result = []
        prev_end = 0

        for tok_idx, (start, end) in enumerate(offsets):
            if end - start != 1:
                # CLS, SEP — skip
                if end > start:
                    prev_end = end
                continue

            # Pass through any characters skipped by the tokenizer
            if start > prev_end:
                result.append(normalized[prev_end:start])

            char = normalized[start:end]
            prev_end = end

            if not _is_hebrew(char):
                if char in ORTHOGRAPHIC_MARKERS:
                    pass
                else:
                    result.append(char)
                continue

            cid = int(consonant_preds[tok_idx])
            allowed = self._letter_constraints.get(char)
            if allowed is not None and cid not in allowed:
                cid = max(allowed, key=lambda x: consonant_logits[0][tok_idx, x])
            consonant = self._consonant_vocab.get(cid, "∅")

            # Geresh rule: if next char is apostrophe, force geresh consonant variant
            if char in self._geresh_map and end < len(normalized) and normalized[end] == "'":
                consonant = self._geresh_map[char]

            vowel = self._vowel_vocab.get(int(vowel_preds[tok_idx]), "∅")
            stress = tok_idx in stressed_positions

            # Assemble IPA chunk: [consonant][ˈ][vowel]
            # Exception: word-final ח with vowel a — furtive patah flips to [ˈ]aχ
            word_final = end >= len(normalized) or not normalized[end].isalpha()
            chunk = ""
            if char == "ח" and word_final and vowel == "a":
                if stress:
                    chunk += STRESS_MARK
                chunk += "aχ"
            else:
                if consonant != "∅":
                    chunk += consonant
                if stress:
                    chunk += STRESS_MARK
                if vowel != "∅":
                    chunk += vowel
            result.append(chunk)

        if prev_end < len(normalized):
            result.append(normalized[prev_end:])

        return "".join(result)
