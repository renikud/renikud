# Architecture

## Problem

Convert unvocalized Hebrew text into IPA. Hebrew is written without vowels — the same string can have multiple valid pronunciations depending on context. The model must recover the pronunciation purely from the consonant skeleton and surrounding context.

## Core Idea

Rather than sequence-to-sequence (which requires alignment at inference time), the model frames G2P as **per-character classification**. Every Hebrew letter independently predicts a `(consonant, vowel, stress)` triple. Non-Hebrew characters (spaces, punctuation, digits, Latin) are passed through unchanged.

This works because Hebrew has a nearly one-to-one letter→phoneme structure: each letter produces exactly one consonant (or silence) and optionally carries a vowel and stress. The model learns the exceptions from context.

## Model

`G2PModel` in `src/model.py`:

1. **Frozen encoder** — `dicta-il/dictabert-large-char-menaked` loaded through Hugging Face `AutoModel` with remote code. `src/encoder.py` returns the bare BERT encoder body, without the nikud prediction heads. Encoder parameters are frozen and kept in eval mode during training.
2. **Trainable context stack** — a 2-layer Transformer encoder on top of DictaBERT hidden states. It uses `d_model=hidden_size`, `nhead=16`, and `dim_feedforward=2816`, for roughly 20M trainable task-specific parameters before the small output classifiers.
3. **Three coupled classification heads** — each head sees the contextual hidden state *plus* the raw logits from the previous head, so later heads have information about earlier predictions rather than being blind to them:
   - **Consonant head** → `hidden` → 26 classes (`∅ b v d h z χ t j k l m n s f p ts tʃ w ʔ ɡ ʁ ʃ ʒ dʒ`)
   - **Vowel head** → `hidden + consonant_logits` → 7 classes (`∅ a e i o u`)
   - **Stress head** → `hidden + consonant_logits + vowel_logits` → 2 classes (none / stressed)
3. **Consonant masking** — logits for phonetically impossible consonants are zeroed out (`-1e9`) using a precomputed per-letter mask from `phonology.py`. For example, ל can only ever produce `l` or `∅`, never `b`.

At inference (`src/infer.py`), the consonant mask is applied before argmax so the model can never predict a phonetically impossible consonant for a given letter — e.g. ק always decodes to `k`, never `v`. Each Hebrew letter position assembles its output as `[consonant][ˈ?][vowel?]`, with one exception: word-final ח with vowel `a` emits `[ˈ?]aχ` (furtive patah — the vowel precedes the consonant in IPA).

## Tokenizer

`src/tokenization.py` loads the `dicta-il/dictabert-large-char` tokenizer with `AutoTokenizer`.

The default BERT pre-tokenizer is replaced with a character-level splitter so each input character gets its own token and a usable `offset_mapping`. This keeps label alignment per character while using DictaBERT's pretrained vocabulary and special-token handling.

## Hebrew Markers

People write Hebrew markers differently (e.g. using English `'`/`"` or Hebrew `׳`/`״`). We normalize all variants to ASCII `'` and `"`, and treat them as orthographic context: they stay in the input and alignment, but their labels are `IGNORE_INDEX`. At decode time they are dropped and cannot win per-word stress.

Example: `בג״ץ` → `בג"ץ` aligns as `ב=ba ג=ɡˈa "=∅ ץ=ts`; labels are `ב=(b,a,0) ג=(ɡ,a,1) "=(IGNORE_INDEX) ץ=(ts,∅,0)`.

## Label Vocabulary

**Consonants** (25 + ∅): `∅ b v d h z χ t j k l m n s f p ts tʃ w ʔ ɡ ʁ ʃ ʒ dʒ`

**Vowels** (6 + ∅): `∅ a e i o u`

**Stress**: binary — 0 (none) or 1 (ˈ precedes vowel)

## Data Pipeline

```
raw TSV (hebrew<TAB>ipa)
  → scripts/prepare_align.py   DP aligner: assigns one IPA chunk per Hebrew letter → JSONL
  → scripts/prepare_tokens.py  tokenize + map labels to token positions → Arrow dataset
  → train.py           training loop
```

The aligner (`src/aligner/align.py`) uses constrained recursive search with memoization to assign one IPA chunk per Hebrew letter. Each letter can only match consonants from its `HEBREW_LETTER_CONSONANTS` entry, which prunes the search space and prevents invalid alignments. `scripts/align.py` parallelizes this across sentences.

Label alignment uses `offset_mapping`: only single-character token positions (offset `end - start == 1`) that correspond to Hebrew letters receive phonological labels. CLS, SEP, spaces, and orthographic markers get `IGNORE_INDEX = -100`. Other punctuation is supervised as silent.
