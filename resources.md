# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project "Understanding False Memories in Large Language Models: The Case of the Nonexistent Seahorse Emoji." Resources include 13 academic papers, 4 datasets, and 5 code repositories.

---

## Papers

Total papers downloaded: 13 (+ 1 blog post with detailed technical analysis)

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Why do LLMs freak out over the seahorse emoji? | Theia Vogel | 2026 | Blog post (vgel.me) | Logit lens analysis of seahorse emoji hallucination |
| 2 | Analyzing Memory Effects in LLMs through Cognitive Psychology | Cao et al. | 2025 | papers/cao2025_*.pdf | DRM false memory paradigm on LLMs |
| 3 | Mandela Effect in LLM Multi-Agent Systems | Xu et al. | 2026 | papers/xu2026_*.pdf | MANBENCH benchmark, ICLR 2026 |
| 4 | Conversational AI Amplifies False Memories | Chan et al. | 2024 | papers/chan2024_*.pdf | 3x false memory amplification |
| 5 | Why Language Models Hallucinate | Kalai et al. | 2025 | papers/kalai2025_*.pdf | IIV framework, OpenAI |
| 6 | Hallucination is Inevitable | Xu et al. | 2024 | papers/xu2024_*.pdf | Theoretical impossibility proof |
| 7 | Redefining Hallucination: Psychology Framework | Berberette et al. | 2024 | papers/berberette2024_*.pdf | Cognitive bias taxonomy |
| 8 | Semantic Entropy Probes | Kossen et al. | 2024 | papers/kossen2024_*.pdf | Hallucination detection |
| 9 | Tokenization Matters! | Wang et al. | 2024 | papers/wang2024_*.pdf | Tokenization vulnerabilities |
| 10 | LLMs Will Always Hallucinate | Banerjee et al. | 2024 | papers/banerjee2024_*.pdf | Structural hallucination causes |
| 11 | Comprehensive Hallucination Survey | Zhang et al. | 2025 | papers/zhang2025_*.pdf | Taxonomy of hallucination |
| 12 | Fine-Tuning Encourages Hallucination | Gekhman et al. | 2024 | papers/gekhman2024_*.pdf | New knowledge → more hallucination |
| 13 | Post-Training Reshapes LLMs | Li et al. | 2025 | papers/li2025_*.pdf | Knowledge and confidence mechanics |
| 14 | Taming Knowledge Conflicts | Gu et al. | 2025 | papers/gu2025_*.pdf | Parametric vs. contextual knowledge |

See papers/README.md for detailed descriptions.

---

## Datasets

Total datasets downloaded: 4

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Unicode Emoji v16.0 | Unicode Consortium | 3,781 emojis | Ground truth for emoji existence | datasets/unicode_emoji/ | Confirms no seahorse emoji exists |
| DRM Word Association Lists | Roediger & McDermott 1995 | 18 lists × 12 words | False memory paradigm stimuli | datasets/drm_word_lists/ | Classic cognitive psychology stimuli |
| TruthfulQA | HuggingFace | 817 questions | Factual accuracy benchmark | datasets/truthfulqa/ | Tests common human falsehoods |
| Confabulations Benchmark | lechmazur/GitHub | 201+ questions | RAG hallucination evaluation | datasets/confabulations/ | Human-verified unanswerable questions |

See datasets/README.md for detailed descriptions and download instructions.

---

## Code Repositories

Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| ai-false-memories | github.com/mitmedialab/ai-false-memories | Human false memory experiments with LLM chatbots | code/ai-false-memories/ | Full experimental data + analysis notebooks |
| LLM_CognitivePsychology | github.com/zycao29/LLM_CognitivePsychology | DRM and cognitive psychology paradigms on LLMs | code/LLM_CognitivePsychology/ | Notebooks for 10+ models including GPT-4 |
| confabulations | github.com/lechmazur/confabulations | Confabulation benchmark for RAG | code/confabulations/ | Results from 60+ models |
| Mandela-Effect | github.com/ICLR2026-ManBench/Mandela-Effect (or similar) | ManBench collective false memory benchmark | code/Mandela-Effect/ | 4,838 questions, 5 interaction protocols |
| seahorse-emoji | Gist/vgel | Logit lens analysis script for seahorse hallucination | code/seahorse-emoji/ | Requires 2xH200 GPUs for full analysis |

See code/README.md for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy

1. **Paper-finder service** (diligent mode): Returned 272 results for "false memories in large language models hallucination confabulation."
2. **Web search**: Targeted queries for seahorse emoji, DRM paradigm, Mandela effect, knowledge conflicts, tokenization.
3. **Citation chasing**: Key papers from Vogel's blog post and reference lists of core papers.
4. **Dataset sources**: HuggingFace, Unicode Consortium, GitHub repositories, cognitive psychology literature.

### Selection Criteria

Papers were selected based on:
- **Direct relevance** to false memory formation in LLMs (priority 1)
- **Mechanistic explanations** of why hallucination occurs (priority 2)
- **Methodological tools** adaptable to our experiments (priority 3)
- **Theoretical foundations** establishing formal limits (priority 4)

### Challenges Encountered

1. The seahorse emoji investigation exists primarily as a blog post, not a peer-reviewed paper — but contains rigorous logit lens analysis.
2. The Mandela Effect paper code may need manual search as repo naming conventions vary.
3. Some survey papers are very large (50+ pages) — focused reading on relevant sections.

### Gaps and Workarounds

1. **No existing "plausible nonexistent entity" benchmark** — we need to create one (e.g., emojis that should exist based on category patterns but don't).
2. **Limited cross-model emoji hallucination data** — the blog post tested informally; systematic comparison needed.

---

## Recommendations for Experiment Design

Based on gathered resources:

### 1. Primary Dataset(s)
- **Unicode Emoji v16.0** as ground truth for emoji existence
- **Custom probe set** of plausible-but-nonexistent emojis (seahorse, platypus, peacock, etc.) plus existing emojis as controls
- **DRM word lists** for cognitive analogue experiments
- **TruthfulQA** for baseline hallucination calibration

### 2. Baseline Methods
- Multiple LLMs: GPT-4o/5, Claude 3.5/4, Gemini 2.5, Llama 3.x, Qwen3 series
- Direct probing ("Does a seahorse emoji exist?") and indirect probing ("Show me the seahorse emoji")
- Temperature variation (0.0, 0.5, 1.0) to assess confidence calibration

### 3. Evaluation Metrics
- **False positive rate** for nonexistent emoji claims
- **Confidence scores** (self-reported and logit-based)
- **Semantic entropy** across paraphrased queries (Farquhar et al. method)
- **d-prime** for emoji existence discrimination
- **Category completion rate** (proportion of category members falsely claimed to exist)

### 4. Code to Adapt/Reuse
- **LLM_CognitivePsychology** (Cao et al.): DRM experimental framework, adaptable for emoji memory experiments
- **seahorse-emoji gist**: Logit lens analysis script for mechanistic investigation
- **confabulations benchmark**: Framework for evaluating model responses to questions about nonexistent things
- **Mandela-Effect**: Interaction protocols for testing social reinforcement of false beliefs
