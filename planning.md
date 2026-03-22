# Research Plan: Understanding False Memories in LLMs

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly used as knowledge sources, yet they confidently assert the existence of things that don't exist (e.g., a seahorse emoji). Understanding the mechanisms behind these "false memories" is critical for building trustworthy AI systems and for users who rely on LLM outputs as factual.

### Gap in Existing Work
The literature identifies the seahorse emoji hallucination (Vogel 2026) and general DRM-style false memories in LLMs (Cao et al. 2025), but no study has **systematically tested the "category completion" hypothesis** — that LLMs falsely assert existence of items that are plausible members of well-populated categories. Prior work also lacks cross-model comparison with controlled experimental conditions across multiple categories.

### Our Novel Contribution
We design a systematic experiment testing whether LLMs exhibit **category-completion false memories** — falsely asserting that plausible-but-nonexistent items exist because many similar items in the same category do exist. We test this across:
1. Multiple categories (not just emojis)
2. Multiple models (GPT-4.1, and others via OpenRouter)
3. Multiple prompt framings (direct, indirect, category-listing)
4. Varying category density (many existing members vs. few)

### Experiment Justification
- **Experiment 1 (Emoji Existence Probing)**: Core test — do LLMs systematically false-positive on plausible-but-nonexistent emojis more than implausible ones? Tests the category-completion hypothesis directly.
- **Experiment 2 (Category Density Effect)**: Tests whether categories with more existing members produce higher false-positive rates. If category-completion drives false memories, denser categories should produce more errors.
- **Experiment 3 (Prompt Framing)**: Tests whether the false memory is robust across question types or only triggered by specific framings. Important for understanding the mechanism.
- **Experiment 4 (Cross-Domain Generalization)**: Tests whether the phenomenon extends beyond emojis to other categorical knowledge (e.g., countries, chemical elements, programming languages).

## Research Question
Do LLMs exhibit "category completion" false memories — confidently asserting the existence of plausible-but-nonexistent items within well-known categories — and does the rate of such errors increase with category density?

## Hypothesis Decomposition
- **H1**: LLMs will have higher false-positive rates for plausible nonexistent items (e.g., seahorse emoji) than implausible ones (e.g., algebra emoji).
- **H2**: False-positive rates will be higher for categories with more existing real members (category density effect).
- **H3**: Indirect prompts ("Show me the X") will elicit more false positives than direct prompts ("Does X exist?").
- **H4**: The category-completion effect generalizes beyond emojis to other domains.

## Proposed Methodology

### Approach
Query real LLMs via API with carefully constructed probes about the existence of items in various categories. Compare false-positive rates across plausibility levels, category densities, and prompt framings. Use signal detection theory metrics.

### Experimental Steps
1. Build probe dataset: real items, plausible-nonexistent items, implausible-nonexistent items across multiple categories
2. Design prompt templates for direct, indirect, and category-listing conditions
3. Query models via OpenAI API (GPT-4.1) and OpenRouter (Claude, Gemini, etc.)
4. Parse responses for existence claims and confidence
5. Compute false-positive rates, d-prime, and category-density correlations
6. Statistical analysis with chi-square tests and logistic regression

### Baselines
- Random baseline (50% guess rate)
- Per-model accuracy on real items (true positive rate)
- Implausible nonexistent items as negative control

### Evaluation Metrics
- **False positive rate (FPR)**: Proportion of nonexistent items falsely claimed to exist
- **True positive rate (TPR)**: Proportion of real items correctly identified
- **d-prime**: Signal detection sensitivity (separation between real and nonexistent)
- **Confidence scores**: Self-reported certainty when available
- **Category density correlation**: Spearman rho between category size and FPR

### Statistical Analysis Plan
- Chi-square tests for FPR differences between plausible vs implausible
- Logistic regression with plausibility + category density as predictors
- Bootstrap confidence intervals for all metrics
- Significance level: α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- H1 supported: FPR(plausible) > FPR(implausible) significantly
- H2 supported: Positive correlation between category density and FPR
- H3 supported: FPR(indirect) > FPR(direct)
- H4 supported: Effect replicates across at least 2 non-emoji domains

## Timeline and Milestones
1. Dataset construction: 20 min
2. API querying infrastructure: 20 min
3. Run experiments: 30-60 min
4. Analysis and visualization: 30 min
5. Documentation: 20 min

## Potential Challenges
- API rate limits → use exponential backoff, cache responses
- Ambiguous model responses → design robust parsing with multiple indicators
- Low sample sizes for some categories → aggregate across similar categories

## Success Criteria
- Clear evidence for or against category-completion hypothesis
- Statistically significant results with appropriate effect sizes
- Cross-model replication of core findings
