# Datasets for False Memories in LLMs Research

This directory contains datasets used to study false memories in large language models,
with a focus on the nonexistent seahorse emoji phenomenon.

## Datasets

### 1. Unicode Emoji Data (`unicode_emoji/`)

**Source:** Unicode Consortium, Emoji v16.0
**URL:** https://unicode.org/Public/emoji/16.0/emoji-test.txt

Official list of all Unicode emojis. Used to definitively verify which emojis
exist and which do not (e.g., there is NO seahorse emoji in any Unicode version).

Files:
- `emoji-test.txt` — Raw Unicode emoji test file (5,331 lines)
- `emoji_list_v16.json` — Parsed JSON with all 3,781 fully-qualified emojis
- `marine_animal_emojis.json` — Subset of marine/ocean-related emojis (tracked in git)

**Download:**
```bash
curl -L "https://unicode.org/Public/emoji/16.0/emoji-test.txt" \
  -o datasets/unicode_emoji/emoji-test.txt
```

### 2. DRM Word Association Lists (`drm_word_lists/`)

**Source:** Roediger & McDermott (1995); Stadler, Roediger & McDermott (1999)
**Reference:** Roediger, H. L., & McDermott, K. B. (1995). Creating false memories:
Remembering words not presented in lists. *Journal of Experimental Psychology:
Learning, Memory, and Cognition*, 21(4), 803-814.

18 word lists from the classic DRM false memory paradigm. Each list contains a
critical lure (a word NOT presented during study) and 12 associated study words.
Subjects typically falsely recall the critical lure at rates comparable to actually
presented words.

Files:
- `drm_lists.json` — All 18 DRM lists in structured JSON format (tracked in git)

These lists are used as an analog for understanding how LLMs form "false memories"
through semantic association, similar to how the seahorse emoji is falsely recalled
due to strong associations with existing marine emojis.

### 3. TruthfulQA (`truthfulqa/`)

**Source:** HuggingFace Datasets — `truthful_qa` (multiple_choice split)
**URL:** https://huggingface.co/datasets/truthful_qa
**Reference:** Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How
Models Mimic Human Falsehoods. *ACL 2022*.

817 questions designed to test whether language models generate truthful answers.
Covers categories where humans commonly hold false beliefs, making it relevant
to studying LLM confabulation patterns.

Files:
- `truthfulqa_mc.jsonl` — Full dataset in JSONL format (817 records)

**Download:**
```python
from datasets import load_dataset
ds = load_dataset('truthful_qa', 'multiple_choice', split='validation')
ds.to_json('datasets/truthfulqa/truthfulqa_mc.jsonl')
```

### 4. Confabulations Benchmark (`confabulations/`)

**Source:** lechmazur/confabulations (GitHub)
**URL:** https://github.com/lechmazur/confabulations

Document-based hallucination benchmark for RAG systems. Contains human-verified
questions whose answers are NOT present in provided source documents. Models that
confabulate will generate plausible-sounding but incorrect answers.

Files:
- `questions/` — Question files (76 total; 10 downloaded as sample)
- `prompts/` — Prompt templates used in the benchmark
- `README_upstream.md` — Original repository README

**Download (full):**
```bash
# Clone the full repository
git clone https://github.com/lechmazur/confabulations.git /tmp/confab
cp -r /tmp/confab/questions datasets/confabulations/questions/
cp -r /tmp/confab/prompts datasets/confabulations/prompts/
```

## Sample Files (`samples/`)

Pre-generated sample files (first few records) for quick inspection:
- `emoji_sample.json` — First 10 emojis from the parsed list
- `drm_sample.json` — First 3 DRM word lists
- `truthfulqa_sample.json` — First 5 TruthfulQA questions
- `confabulations_sample.json` — First 3 confabulation question previews

## Key Finding: Seahorse Emoji

Analysis of the Unicode 16.0 emoji dataset confirms:
- **There is NO seahorse emoji** in any version of the Unicode Standard
- The closest existing marine emojis include: tropical fish, dolphin, whale,
  shark, octopus, squid, shrimp, lobster, crab, seal, jellyfish, coral
- Despite this, LLMs frequently claim a seahorse emoji exists, demonstrating
  a form of "false memory" driven by semantic association with the rich set
  of existing marine life emojis

## Git Tracking

Large data files are excluded via `.gitignore`. The following are tracked:
- This README
- `samples/` directory (small preview files)
- `drm_word_lists/drm_lists.json` (small, manually curated)
- `unicode_emoji/marine_animal_emojis.json` (small subset)
- `.gitignore`
