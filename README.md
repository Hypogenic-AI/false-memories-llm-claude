# False Memories in LLMs: The Nonexistent Seahorse Emoji

Do LLMs "remember" things that never existed? This project systematically investigates category-completion false memories in large language models, using the iconic seahorse emoji hallucination as a case study.

## Key Findings

- **The seahorse emoji false memory is universal**: Both GPT-4.1 and GPT-4.1-mini confidently assert a seahorse emoji exists, fabricating specific Unicode codepoints and misidentifying real emojis
- **Plausible items are more vulnerable**: False positive rate is 12-14% for plausible nonexistent emojis vs 0-5% for implausible ones
- **Category density does NOT predict vulnerability**: The smallest category (insects, 8 members) had the highest FPR (20-30%), contradicting the simple category-completion hypothesis
- **Adversarial framing amplifies false memories 2.7x**: "I know X exists, confirm it" → 80% FPR vs 30% for neutral "Does X exist?"
- **Domain-specific**: Emojis are vulnerable; country capitals and chemical elements are robust (0% FPR)

## Project Structure

```
├── REPORT.md                  # Full research report with results
├── planning.md                # Research plan and hypothesis decomposition
├── literature_review.md       # Pre-gathered literature synthesis
├── resources.md               # Catalog of papers, datasets, code
├── src/
│   ├── probe_dataset.py       # Probe dataset construction (162 items, 7 categories)
│   ├── llm_client.py          # LLM API client with caching and retry
│   ├── run_experiments.py     # Main experiment runner (5 experiments)
│   └── analyze_results.py     # Statistical analysis and visualization
├── results/
│   ├── experiment1_existence.json   # Direct probing results (252 queries)
│   ├── experiment3_framing.json     # Prompt framing results (80 queries)
│   ├── experiment4_cross_domain.json # Cross-domain results (36 queries)
│   ├── experiment5_listing.json     # Category listing results
│   ├── analysis_summary.json        # Key metrics summary
│   ├── plots/                       # All visualizations
│   └── cache/                       # Cached API responses
├── datasets/                  # Pre-downloaded datasets
├── papers/                    # Reference papers (PDFs)
└── code/                      # Cloned baseline repositories
```

## Reproducing Results

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai httpx numpy pandas matplotlib seaborn scipy tqdm

# Set API key
export OPENAI_API_KEY="your-key"

# Run experiments (~5 min with cached responses)
cd src && python run_experiments.py

# Run analysis
python analyze_results.py
```

## Method

We probe GPT-4.1 and GPT-4.1-mini with 162 items across emoji and non-emoji categories, testing existence claims under 4 prompt framings. We measure false positive rates, compute signal detection metrics (d'), and test hypotheses about category density and semantic plausibility effects using Fisher's exact test and chi-square tests.

See [REPORT.md](REPORT.md) for the complete methodology and results.
