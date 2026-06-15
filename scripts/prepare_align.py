import argparse
import json
import regex as re
import unicodedata
import os
import sys
import concurrent.futures
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aligner.align import align_word

from phonology import ORTHOGRAPHIC_MARKERS, normalize_graphemes

# Compiled regex is faster
MARKERS = "".join(ORTHOGRAPHIC_MARKERS)
HEB_RE = re.compile(rf"[^\u05d0-\u05ea{re.escape(MARKERS)}]")
IPA_RE = re.compile(r"[^abdefghijklmnoprstuvwzɡʁʃʒʔˈχ]")
NIKUD_RE = re.compile(r"[\p{M}|]")
BIDI_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")

def normalize(text: str) -> str:
    text = NIKUD_RE.sub("", unicodedata.normalize("NFD", text))
    text = BIDI_RE.sub("", text)
    text = normalize_graphemes(text)
    # Treat hyphens as word boundaries
    text = text.replace("-", " ")
    return text

def align_sentence(heb: str, ipa: str) -> list[tuple[str, str]] | None:
    heb_words, ipa_words = heb.split(), ipa.split()
    if len(heb_words) != len(ipa_words):
        return None

    result = []
    for hw, iw in zip(heb_words, ipa_words):
        heb_core = HEB_RE.sub("", hw)
        ipa_core = IPA_RE.sub("", iw)
        if not heb_core:
            continue
        
        aligned = align_word(heb_core, ipa_core)
        if aligned is None:
            return None
        
        if result:
            result.append((" ", " "))
        result.extend(aligned)
    return result

def process_line(line: str) -> str | tuple[str, str]:
    """Processes a single line and returns the JSON string or failure info."""
    heb_raw, sep, ipa = line.strip().partition("\t")
    if not sep:
        return "FAIL_EMPTY"
    
    # Treat hyphens as word boundaries in IPA as well to match Hebrew normalization
    ipa = ipa.replace("-", " ")
    
    heb = normalize(heb_raw)
    result = align_sentence(heb, ipa)
    
    if result is not None:
        # Return serialized JSON directly to avoid pickling complex objects
        return ("SUCCESS", json.dumps({"hebrew": heb, "alignment": result, "phonemes": ipa}, ensure_ascii=False))
    else:
        return ("FAIL", f"{heb}\t{ipa}")

def main():
    parser = argparse.ArgumentParser(description="High-speed Aligner")
    parser.add_argument("input", help="Input TSV")
    parser.add_argument("output", help="Output JSONL")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--chunk_size", type=int, default=1000) # Balanced for responsiveness
    args = parser.parse_args()

    failures_path = args.output.replace(".jsonl", "_failures.txt")
    
    aligned_count = 0
    failed_count = 0

    # Use a ProcessPoolExecutor for heavy CPU tasks
    # In Python 3.14, this is highly optimized
    with (
        open(args.input, "r", encoding="utf-8") as fin,
        open(args.output, "w", encoding="utf-8") as fout,
        open(failures_path, "w", encoding="utf-8") as ffail,
        concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor
    ):
        # We use a generator to stream lines so we don't load the file into RAM
        # chunksize=1000 keeps the workers busy without overwhelming the queue
        results = executor.map(process_line, fin, chunksize=args.chunk_size)
        
        # tqdm now shows real progress because we are streaming
        # Note: We don't have a 'total' unless we count lines first, 
        # but the responsiveness is instant.
        for status_info in tqdm(results, desc="Aligning"):
            if status_info == "FAIL_EMPTY":
                continue
            
            status, data = status_info
            if status == "SUCCESS":
                fout.write(data + "\n")
                aligned_count += 1
            else:
                ffail.write(data + "\n")
                failed_count += 1

    print(f"\nAligned: {aligned_count:,} | Failed: {failed_count:,}")

if __name__ == "__main__":
    main()
