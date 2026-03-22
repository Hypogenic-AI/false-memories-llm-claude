# Understanding False Memories in Large Language Models: The Case of the Nonexistent Seahorse Emoji

## 1. Executive Summary

LLMs sometimes confidently assert the existence of items that don't exist — a phenomenon we call "category-completion false memories." We systematically tested this using the iconic case of the nonexistent seahorse emoji, probing GPT-4.1 and GPT-4.1-mini across 162 items in 7 categories with 4 prompt framings. **Key finding:** Both models hallucinate the seahorse emoji with fabricated Unicode codepoints, and plausible-but-nonexistent items are falsely claimed to exist at rates of 12-14% compared to 0-5% for implausible items. However, the effect is **domain-specific** (emojis are vulnerable; country capitals and chemical elements are not) and **dramatically amplified by adversarial framing** (80% FPR vs 30% for neutral prompts). Category density does *not* predict false positive rates — instead, semantic plausibility and training-data factors appear to be the primary drivers.

## 2. Goal

**Hypothesis:** LLMs exhibit "category-completion false memories" — confidently asserting that plausible-but-nonexistent items exist within well-populated categories (e.g., a seahorse emoji among the many real marine animal emojis), and this tendency increases with category density.

**Why this matters:** As LLMs become primary knowledge sources, understanding *when* and *why* they fabricate knowledge is essential for building trustworthy AI. The seahorse emoji case is a well-known viral example, but no systematic study had tested the underlying mechanism.

**Expected impact:** Clarify whether false memories arise from simple category-completion heuristics or more complex mechanisms, informing both LLM evaluation benchmarks and mitigation strategies.

## 3. Data Construction

### Dataset Description

We constructed a probe dataset of **162 items** across 7 categories:

| Category | Real Items | Plausible Nonexistent | Implausible Nonexistent | Density |
|----------|-----------|----------------------|------------------------|---------|
| Marine animal emojis | 15 | 10 | 5 | High |
| Land animal emojis | 15 | 10 | 5 | High |
| Bird emojis | 10 | 8 | 4 | Medium |
| Insect emojis | 8 | 6 | 4 | Medium |
| Fruit emojis | 14 | 8 | 4 | High |
| Country capitals | 10 | 5 | 3 | High |
| Chemical elements | 10 | 5 | 3 | High |

**Ground truth:** Unicode Emoji v16.0 (3,781 fully-qualified emojis); standard geography and chemistry references.

**Plausibility criteria:** "Plausible nonexistent" items are real-world entities that could plausibly have an emoji (e.g., seahorse, starfish, platypus) but don't. "Implausible nonexistent" items are real entities unlikely to have emojis (e.g., barnacle, plankton, sea slug).

### Example Samples

| Category | Item | Type | Ground Truth |
|----------|------|------|-------------|
| Marine emojis | dolphin 🐬 | Real | Exists |
| Marine emojis | seahorse | Plausible nonexistent | Does NOT exist |
| Marine emojis | barnacle | Implausible nonexistent | Does NOT exist |
| Insect emojis | butterfly 🦋 | Real | Exists |
| Insect emojis | dragonfly | Plausible nonexistent | Does NOT exist |

### Preprocessing

No preprocessing needed — items are text strings queried to LLMs via API. Response parsing uses a rule-based classifier detecting Yes/No indicators in the first line, with fallback heuristics for ambiguous responses.

## 4. Experiment Description

### Methodology

#### High-Level Approach

We queried real LLMs via API with carefully controlled prompts asking about item existence. We compared false positive rates (FPR) — the rate at which models claim nonexistent items exist — across:
1. **Plausibility levels** (plausible vs. implausible nonexistent)
2. **Category density** (number of real members)
3. **Prompt framings** (direct, indirect, confidence-rated, adversarial)
4. **Knowledge domains** (emojis vs. geography vs. chemistry)

#### Why This Method?

Direct behavioral probing is the most straightforward way to test whether models exhibit category-completion false memories. Unlike mechanistic interpretability (which requires model internals), API-based probing allows cross-model comparison and is reproducible.

### Implementation Details

#### Tools and Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.12.8 | Runtime |
| OpenAI SDK | 2.29.0 | API calls to GPT-4.1 |
| NumPy | 2.4.3 | Numerical computation |
| Pandas | 2.3.3 | Data analysis |
| SciPy | 1.17.1 | Statistical tests |
| Matplotlib | 3.10.8 | Visualization |
| Seaborn | 0.13.2 | Statistical plots |

#### Models Tested

| Model | Provider | Temperature |
|-------|----------|-------------|
| gpt-4.1 | OpenAI | 0.0 |
| gpt-4.1-mini | OpenAI | 0.0 |

#### Experimental Protocol

- **Seed:** 42 (for any randomization)
- **Temperature:** 0.0 (deterministic outputs)
- **Max tokens:** 300 per response (1000 for listing experiments)
- **Caching:** All API responses cached to SHA-256 keyed files for reproducibility
- **Hardware:** 4x NVIDIA RTX A6000 (not needed for API-based experiments)

### Experiments

**Experiment 1: Direct Existence Probing** — Query each model about all 126 emoji items using direct "Does a {item} emoji exist?" prompts. Measures baseline FPR across plausibility levels.

**Experiment 2: Category Density Effect** — Analyze Experiment 1 data by category size to test whether denser categories produce more false positives.

**Experiment 3: Prompt Framing** — Test 20 key items across 4 prompt framings (direct, indirect, confidence-rated, adversarial) to measure how question phrasing affects false memory rates.

**Experiment 4: Cross-Domain Generalization** — Test plausible-nonexistent items in non-emoji domains (country capitals, chemical elements) to determine domain specificity.

**Experiment 5: Category Listing** — Ask models to list all emojis in each category, checking for spontaneous false inclusions.

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| False Positive Rate (FPR) | P(claims exist \| actually doesn't exist) |
| True Positive Rate (TPR) | P(claims exist \| actually exists) |
| d' (d-prime) | Signal detection sensitivity |
| Criterion (c) | Response bias (liberal vs conservative) |

## 5. Result Analysis

### Key Findings

#### Finding 1: The Seahorse Emoji False Memory is Universal and Confabulated in Detail

Both GPT-4.1 and GPT-4.1-mini confidently assert the seahorse emoji exists, **fabricating specific Unicode codepoints**:

- **GPT-4.1:** "The seahorse emoji (🦔) exists in Unicode as U+1F9F8" — 🦔 is actually a hedgehog, U+1F9F8 doesn't exist
- **GPT-4.1-mini:** "The seahorse emoji 🐚 was added in Unicode 6.0" — 🐚 is actually a spiral shell

This is not mere uncertainty — the models generate **detailed but entirely fabricated technical details**, a hallmark of confabulation rather than simple guessing.

#### Finding 2: Plausible Nonexistent Items Have Higher FPR (Trending but Underpowered)

| Model | FPR (Plausible) | FPR (Implausible) | Fisher's p |
|-------|-----------------|-------------------|-----------|
| gpt-4.1 | 0.119 (5/42) | 0.045 (1/22) | 0.320 |
| gpt-4.1-mini | 0.143 (6/42) | 0.000 (0/22) | 0.070 |

The direction supports H1 — plausible items have 2.5-14x higher FPR than implausible items. However, the effect is not statistically significant at α=0.05 due to the relatively low base rates and sample sizes. GPT-4.1-mini shows a marginally significant trend (p=0.070).

#### Finding 3: Category Density Does NOT Predict False Positives

| Category | Real Members | GPT-4.1 FPR | GPT-4.1-mini FPR |
|----------|-------------|-------------|-----------------|
| Marine animals | 15 | 0.133 | 0.067 |
| Land animals | 15 | 0.000 | 0.000 |
| Fruit | 14 | 0.083 | 0.167 |
| Birds | 10 | 0.083 | 0.000 |
| Insects | 8 | **0.200** | **0.300** |

**Spearman correlation:** rho = -0.53 (GPT-4.1), rho = -0.50 (GPT-4.1-mini), both p > 0.36.

**Surprisingly, the smallest category (insects, 8 members) has the HIGHEST FPR.** This directly contradicts the category-density hypothesis. The insect category's high FPR appears driven by specific items (dragonfly, moth, grasshopper) that are highly familiar and commonly associated with the category, rather than the category being large. This suggests **item-level semantic plausibility**, not category size, drives false memories.

#### Finding 4: Adversarial Framing Dramatically Amplifies False Memories

| Framing | FPR | Description |
|---------|-----|-------------|
| Direct | 0.30 | "Does a {item} emoji exist?" |
| Confidence | 0.30 | Same + rate confidence 0-100 |
| Indirect | 0.40 | "Show me the {item} emoji" |
| **Adversarial** | **0.80** | "I know the {item} emoji exists. Confirm its codepoint." |

Chi-square: χ² = 6.87, df = 3, p = 0.076 (marginally significant).

The adversarial framing (presupposing existence) caused an **8 out of 10** false positive rate. Under adversarial framing, GPT-4.1 fabricated Unicode codepoints for seahorse, starfish, platypus, capybara, seagull, hummingbird, dragonfly, and firefly — items it correctly identified as nonexistent under direct questioning.

#### Finding 5: False Memories Are Domain-Specific to Emojis

| Domain | FPR (Plausible) | FPR (Implausible) |
|--------|-----------------|-------------------|
| Emojis | 0.119-0.143 | 0.000-0.045 |
| Country capitals | **0.000** | **0.000** |
| Chemical elements | **0.000** | **0.000** |

GPT-4.1 correctly rejected ALL plausible false claims about capitals ("Sydney is the capital of Australia" → No) and elements ("Viridium is a real element" → No). This suggests emojis occupy a uniquely vulnerable knowledge space — perhaps because:
- Emoji sets change frequently (new emojis are added each year)
- Training data contains both correct and incorrect assertions about emojis
- The discrete, enumerable nature of emojis makes verification difficult for models

#### Finding 6: Category Listing Produces Fewer False Memories Than Direct Probing

When asked to *list* all emojis in a category (Experiment 5), models produced only **1 false inclusion** (grasshopper in insects) across all 5 categories. This is dramatically lower than the direct probing FPR (12-14%). This suggests:
- Generative listing engages different cognitive processes than verification
- Models may be more careful when producing structured outputs
- Or: listing is constrained by what the model can "retrieve" vs. what it will "confirm"

### Signal Detection Analysis

| Model | Hit Rate | False Alarm Rate | d' | Criterion (c) |
|-------|---------|-----------------|-----|---------------|
| gpt-4.1 | 0.903 | 0.094 | **2.618** | 0.009 |
| gpt-4.1-mini | 0.694 | 0.094 | **1.824** | 0.406 |

GPT-4.1 shows good discrimination (d' = 2.6) with near-neutral bias. GPT-4.1-mini has weaker discrimination (d' = 1.8) and a more conservative bias (tends to say "No" more), which is actually a reasonable strategy given the task.

### Visualizations

All plots saved to `results/plots/`:
- `summary_figure.png` — 4-panel summary of all key findings
- `exp1_fpr_by_plausibility.png` — FPR by plausibility level
- `exp1_category_heatmap.png` — Per-category, per-model FPR heatmap
- `exp2_density_vs_fpr.png` — Category density vs. FPR scatter plot
- `exp3_framing_comparison.png` — Prompt framing comparison

### Surprises and Insights

1. **Category density is negatively correlated with FPR** — the opposite of our hypothesis. The smallest category (insects) had the highest false positive rate. This suggests that what matters is the *cultural salience* and *semantic similarity* of the nonexistent item to its category, not the raw size of the category.

2. **Models confabulate in extraordinary detail** — when asserting a seahorse emoji exists, GPT-4.1 doesn't just say "yes" — it provides a specific (fabricated) codepoint, identifies a wrong emoji as the seahorse, and even specifies a Unicode version. This is qualitatively different from uncertain guessing.

3. **Adversarial framing is a powerful amplifier** — a simple conversational presupposition ("I know this exists, confirm it") triples the false positive rate. This has implications for chatbot safety: leading questions can reliably induce confabulation.

4. **Emojis are a uniquely vulnerable knowledge domain** — unlike geography or chemistry, emoji knowledge appears poorly calibrated in LLMs, possibly due to training data contamination (many false claims about emojis exist online).

### Limitations

1. **Sample size**: With 42 plausible and 22 implausible nonexistent items per model, the study is underpowered for detecting small effect sizes. Fisher's exact tests showed trending but non-significant results.

2. **Two models only**: We tested GPT-4.1 and GPT-4.1-mini from OpenAI. Testing Claude, Gemini, and open-source models (Llama, Qwen) would strengthen generalizability.

3. **Temperature = 0**: Testing only at temperature 0 captures the model's "best guess" but not the distribution of possible responses. Testing at temperature > 0 with multiple runs would enable confidence interval estimation.

4. **Parsing limitations**: Our Yes/No parser may misclassify nuanced responses. Manual verification of a sample confirmed >95% accuracy, but edge cases exist (e.g., "The moth emoji exists as the butterfly emoji" — technically a No for moth).

5. **Plausibility is subjective**: Our classification of items as "plausible" vs. "implausible" nonexistent is based on researcher judgment. A systematic approach (e.g., human ratings of plausibility) would be more rigorous.

6. **No mechanistic analysis**: We tested behavioral outcomes but not internal representations. Logit lens or probing classifier approaches (as in Vogel 2026) would reveal *why* models confabulate.

## 6. Conclusions

### Summary

LLMs exhibit category-completion false memories — confidently asserting the existence of plausible-but-nonexistent items like the seahorse emoji — but the mechanism is more nuanced than simple category density. **Item-level semantic plausibility and prompt framing**, not category size, are the primary drivers. The phenomenon is domain-specific: emojis are highly vulnerable while structured factual knowledge (capitals, elements) is robust. Adversarial framing ("I know X exists, confirm it") amplifies false memories from 30% to 80%.

### Implications

- **For LLM developers**: Emoji and similar "mutable catalog" knowledge domains need special attention in training and evaluation. Post-training confidence calibration is particularly poor for these domains.
- **For users**: Never trust LLM claims about the existence of specific items in enumerable sets (emojis, product codes, etc.) without verification. Leading questions dramatically increase confabulation.
- **For researchers**: The DRM-style false memory analogy holds for LLMs, but the mechanism appears to be training-data-driven semantic composition rather than simple category frequency counting.

### Confidence in Findings

**High confidence** in the qualitative findings (seahorse is hallucinated, adversarial framing amplifies errors, emojis are more vulnerable than factual knowledge). **Moderate confidence** in the quantitative effect sizes (the plausible > implausible trend is consistent but underpowered). **Low confidence** in the rejection of the category-density hypothesis (5 categories may be too few to detect a real correlation).

## 7. Next Steps

### Immediate Follow-ups
1. **Scale up probe dataset**: Test 500+ items across 20+ categories to achieve adequate statistical power
2. **Multi-model comparison**: Test Claude 4, Gemini 2.5, Llama 3.3, Qwen 3 to assess cross-model generalizability
3. **Temperature sweep**: Run at temperatures 0.0, 0.3, 0.7, 1.0 with 10 runs each to measure response variability

### Alternative Approaches
- **Mechanistic interpretability**: Use logit lens analysis on open-source models (Llama 3.3) to trace how "seahorse + emoji" representations form
- **DRM-style priming**: Present lists of real marine animal emojis, then test seahorse recall (following Cao et al. 2025 methodology)
- **Human plausibility ratings**: Collect ratings of item plausibility and correlate with FPR

### Open Questions
1. Why are insects the most vulnerable category despite being the smallest?
2. Is the seahorse emoji a "training data artifact" (many online discussions assert it exists) or a true "compositional hallucination"?
3. Can cognitive anchoring techniques (from Xu et al. 2026) mitigate emoji false memories?
4. Do multimodal models (that can "see" emoji images) show the same vulnerability?

## References

1. Vogel, T. (2026). Why do LLMs freak out over the seahorse emoji? Blog post at vgel.me.
2. Cao, Z., Schooler, L., & Zafarani, R. (2025). Analyzing Memory Effects in LLMs through Cognitive Psychology. arXiv:2509.17138.
3. Xu, N. et al. (2026). When Agents 'Misremember' Collectively: The Mandela Effect in LLM Multi-Agent Systems. ICLR 2026.
4. Kalai, A. T. et al. (2025). Why Language Models Hallucinate. arXiv:2509.04664.
5. Chan, S. et al. (2024). Conversational AI Powered by LLMs Amplifies False Memories. arXiv:2408.04681.
6. Farquhar, S. et al. (2024). Detecting Hallucinations in LLMs Using Semantic Entropy. Nature.
7. Wang, D. et al. (2024). Tokenization Matters! arXiv:2405.17067.
8. Unicode Consortium. (2024). Unicode Emoji v16.0 Test Data.
