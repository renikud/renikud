"""
Extract maximal aligned Hebrew-IPA subsequences from normalized two-column TSV.

Input format:
    hebrew<TAB>ipa

The extractor cleans punctuation from IPA tokens before calling the aligner.
This matters because ASR output often has tokens like "hazˈe." or "ʁˈoʃ,"
that are otherwise valid pronunciations.

Usage:
    uv run scripts/extract_pairs_clean.py \
      dataset/voxknesset-whisper-abjad-he-ipa-normalized.tsv \
      dataset/voxknesset-whisper-abjad-he-ipa-pairs-clean.tsv
"""

from __future__ import annotations

import argparse
import functools
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aligner.align import align_word  # noqa: E402


HEBREW_RE = re.compile(r"[\u05d0-\u05ea]")
HEBREW_STRIP_RE = re.compile(r"[^ \u05d0-\u05ea'\"]+")
IPA_STRIP_RE = re.compile(r"[^a-zɡʔʁχʃʒˈ]+")


def normalize_hebrew_text(text: str) -> str:
    text = re.sub(r"[׳`´]", "'", text)
    text = re.sub(r'[״“”]', '"', text)
    text = text.replace("-", " ")
    text = HEBREW_STRIP_RE.sub(" ", text)
    return " ".join(text.split())


def clean_ipa_token(token: str) -> str:
    return IPA_STRIP_RE.sub("", token)


def normalize_ipa_text(text: str) -> str:
    text = text.replace("-", " ")
    return " ".join(t for t in (clean_ipa_token(tok) for tok in text.split()) if t)


@functools.lru_cache(maxsize=1_000_000)
def try_align(heb_word: str, ipa_word: str) -> bool:
    return bool(heb_word) and bool(ipa_word) and align_word(heb_word, ipa_word) is not None


def flush_sequence(
    sequences: list[tuple[str, str]],
    heb_words: list[str],
    ipa_words: list[str],
    min_words: int,
) -> None:
    if len(heb_words) >= min_words:
        sequences.append((" ".join(heb_words), " ".join(ipa_words)))


def extract_sequences(
    heb: str,
    ipa: str,
    *,
    lookahead: int,
    min_words: int,
) -> tuple[list[tuple[str, str]], int, int, int, int]:
    heb_words = normalize_hebrew_text(heb).split()
    ipa_words = normalize_ipa_text(ipa).split()

    sequences: list[tuple[str, str]] = []
    current_heb: list[str] = []
    current_ipa: list[str] = []
    matched_words = 0
    skipped_heb = 0
    skipped_ipa = 0

    hi, pi = 0, 0
    while hi < len(heb_words) and pi < len(ipa_words):
        hw = heb_words[hi]
        pw = ipa_words[pi]

        if not HEBREW_RE.search(hw):
            flush_sequence(sequences, current_heb, current_ipa, min_words)
            current_heb, current_ipa = [], []
            hi += 1
            continue

        if try_align(hw, pw):
            current_heb.append(hw)
            current_ipa.append(pw)
            matched_words += 1
            hi += 1
            pi += 1
            continue

        resynced = False
        for skip in range(1, lookahead + 1):
            if hi + skip < len(heb_words) and try_align(heb_words[hi + skip], pw):
                flush_sequence(sequences, current_heb, current_ipa, min_words)
                current_heb, current_ipa = [], []
                skipped_heb += skip
                hi += skip
                resynced = True
                break

        if not resynced:
            for skip in range(1, lookahead + 1):
                if pi + skip < len(ipa_words) and try_align(hw, ipa_words[pi + skip]):
                    flush_sequence(sequences, current_heb, current_ipa, min_words)
                    current_heb, current_ipa = [], []
                    skipped_ipa += skip
                    pi += skip
                    resynced = True
                    break

        if not resynced:
            flush_sequence(sequences, current_heb, current_ipa, min_words)
            current_heb, current_ipa = [], []
            skipped_heb += 1
            skipped_ipa += 1
            hi += 1
            pi += 1

    flush_sequence(sequences, current_heb, current_ipa, min_words)
    return sequences, len(heb_words), len(ipa_words), matched_words, skipped_heb + skipped_ipa


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    total = count_lines(args.input)
    if args.limit is not None:
        total = min(total, args.limit)

    rows = 0
    emitted = 0
    emitted_words = 0
    total_heb_words = 0
    total_ipa_words = 0
    matched_words = 0
    skipped_items = 0

    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=total, desc="extract clean pairs"):
            if args.limit is not None and rows >= args.limit:
                break
            rows += 1
            heb, sep, ipa = line.rstrip("\n").partition("\t")
            if not sep:
                continue

            seqs, heb_count, ipa_count, match_count, skip_count = extract_sequences(
                heb,
                ipa,
                lookahead=args.lookahead,
                min_words=args.min_words,
            )
            total_heb_words += heb_count
            total_ipa_words += ipa_count
            matched_words += match_count
            skipped_items += skip_count

            for seq_heb, seq_ipa in seqs:
                fout.write(f"{seq_heb}\t{seq_ipa}\n")
                emitted += 1
                emitted_words += len(seq_heb.split())

            if rows % 1000 == 0:
                tqdm.write(
                    "rows={:,} matched={:.2f}% emitted={:,} emitted_words={:,}".format(
                        rows,
                        100 * matched_words / total_heb_words if total_heb_words else 0.0,
                        emitted,
                        emitted_words,
                    )
                )

    print(f"rows: {rows:,}")
    print(f"hebrew words: {total_heb_words:,}")
    print(f"ipa words: {total_ipa_words:,}")
    print(f"matched words: {matched_words:,} ({100 * matched_words / total_heb_words:.2f}%)")
    print(f"emitted sequences: {emitted:,}")
    print(f"emitted sequence words: {emitted_words:,} ({100 * emitted_words / total_heb_words:.2f}%)")
    print(f"skipped alignment items: {skipped_items:,}")
    print(f"align cache: {try_align.cache_info()}")


if __name__ == "__main__":
    main()
