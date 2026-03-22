# Literature Review: Understanding False Memories in Large Language Models

## Research Area Overview

This review examines the intersection of LLM hallucination/confabulation and cognitive psychology's false memory paradigms, motivated by the specific case of the nonexistent seahorse emoji — a phenomenon where LLMs confidently assert the existence of a Unicode emoji that has never existed. The research spans mechanistic interpretability, cognitive analogues of LLM behavior, knowledge conflict resolution, and collective memory distortion in multi-agent systems.

---

## Key Papers

### 1. "Why do LLMs freak out over the seahorse emoji?" (Theia Vogel, 2026)
- **Source**: Blog post at vgel.me/posts/seahorse/
- **Key Contribution**: First detailed mechanistic analysis of the seahorse emoji hallucination using logit lens interpretability.
- **Models Tested**: GPT-5 Instant, GPT-5, Claude Sonnet 4.5, Gemini 2.5 Pro, Llama 3.3 70B (all showed 83-100% confidence that the emoji exists).
- **Methodology**: Applied logit lens (output projection at intermediate layers) to trace how models construct the concept "seahorse + emoji" through layers, finding the model builds a valid embedding-space representation that has no corresponding output token.
- **Key Findings**: (1) Models compose "seahorse + emoji" via word2vec-style vector addition; (2) Middle layers (52-72) refine toward emoji byte patterns (e.g., 'ĠðŁĲł'); (3) The model only discovers the token doesn't exist at the final lm_head projection; (4) Training data contains both correct ("no seahorse emoji") and false ("I remember this emoji") statements, creating ambiguity.
- **Relevance**: Directly addresses our research hypothesis — the hallucination arises from semantic composition in embedding space combined with the absence of negative examples.

### 2. "Analyzing Memory Effects in LLMs through the Lens of Cognitive Psychology" (Cao et al., 2025)
- **Authors**: Zhaoyang Cao, Lael Schooler, Reza Zafarani (Syracuse University)
- **Year**: 2025 (arXiv:2509.17138)
- **Key Contribution**: Systematically tests seven human memory phenomena (from Schacter's "Seven Sins of Memory") on LLMs, including the DRM false memory paradigm.
- **Models Tested**: GPT-4, Mistral-7B-Instruct, LLaMA-3-8B
- **Datasets**: Person-location pairs (from Schneider & Anderson), DRM word association lists (from Roediger & McDermott 1995, 12 target words × 15 associates each).
- **Key Results**:
  - LLMs exhibit DRM false memories: critical lure recall rate = 11.4% (items never presented but semantically related).
  - Fan effect confirmed: accuracy drops from 0.991 (Fan 1) to 0.915 (Fan 2) as associations increase.
  - List length effect confirmed: accuracy declines with more items (0.99 → 0.91).
  - Divergences: LLMs are position-invariant (no primacy/recency) and robust to nonsense stimuli.
- **Evaluation Metrics**: Recall accuracy, false alarm rate, hit rate, d-prime (signal detection theory).
- **Code**: https://github.com/zycao29/LLM_CognitivePsychology.git
- **Relevance**: Provides direct experimental evidence that LLMs form false memories via semantic activation — the same mechanism that could cause a seahorse emoji confabulation (many marine animal emojis exist, creating high associative fan).

### 3. "When Agents 'Misremember' Collectively: Exploring the Mandela Effect in LLM-Based Multi-Agent Systems" (Xu et al., ICLR 2026)
- **Authors**: Naen Xu et al. (Zhejiang University, UCLA, Palo Alto Networks)
- **Year**: 2026 (arXiv:2602.00428)
- **Key Contribution**: Introduces MANBENCH benchmark (4,838 questions) for evaluating collective false memory in multi-agent LLM systems.
- **Models Tested**: 13 LLMs (GPT-4o/5, Claude 3.5/4, Gemini 2.5, Llama 3.x, Deepseek, Qwen3 series).
- **Key Results**:
  - All LLMs are susceptible; GPT-5's error rate doubled (17.6% → 41.6%) under social influence.
  - Role-based false narratives more potent than simple consensus.
  - Inverse scaling in Qwen3: larger models (235B) *more* susceptible than smaller ones (8B).
  - Long-term memory consolidation: some models retain false beliefs persistently (Claude 3.5 Haiku: 55.6% retention).
  - Mitigations: Cognitive anchoring achieved avg 74.4% reduction; balanced SFT reduced Llama3's shift from 99.5% → 21.5%.
- **Evaluation Metrics**: Error rate, reality shift rate, maximal reality shift rate.
- **Relevance**: Shows false memories can be socially reinforced and consolidated — relevant to how training data consensus on nonexistent things can create persistent model beliefs.

### 4. "Why Language Models Hallucinate" (Kalai, Nachum, Vempala, Zhang, 2025)
- **Authors**: OpenAI and Georgia Tech
- **Year**: 2025 (arXiv:2509.04664)
- **Key Contribution**: Formal theoretical framework (Is-It-Valid reduction) proving hallucination is statistically inevitable and evaluation design perpetuates it.
- **Key Findings**:
  - Generative error rate ≥ 2 × IIV misclassification rate (formal lower bound).
  - Singleton rate in training data directly predicts hallucination rate on arbitrary facts.
  - 9/10 major benchmarks use binary grading that rewards guessing over abstaining.
  - Post-training destroys calibration (ECE: 0.007 → 0.074), making models overconfident.
- **Relevance**: Explains why LLMs produce confident false assertions rather than expressing uncertainty — directly applicable to the seahorse emoji phenomenon.

### 5. "Conversational AI Powered by LLMs Amplifies False Memories in Witness Interviews" (Chan et al., 2024)
- **Authors**: Samantha Chan et al. (MIT Media Lab, UC Irvine)
- **Year**: 2024 (arXiv:2408.04681)
- **Key Contribution**: Shows GPT-4-powered chatbots induce 3x more false memories in humans than control conditions.
- **Models Used**: GPT-4 (generative chatbot condition).
- **Datasets**: Custom crime witness experiment with 200 participants, 5 misleading questions.
- **Key Results**: Generative chatbot → 36.4% false memory rate; effects persist 1 week; sycophancy and elaborative confabulation are key mechanisms.
- **Code**: https://github.com/mitmedialab/ai-false-memories
- **Relevance**: Demonstrates the downstream impact — when LLMs confabulate (e.g., asserting seahorse emoji exists), human users can encode these as persistent false memories.

### 6. "Hallucination is Inevitable: An Innate Limitation of LLMs" (Xu, Jain, Kankanhalli, 2024)
- **Authors**: Ziwei Xu et al. (National University of Singapore)
- **Year**: 2024 (arXiv:2401.11817)
- **Key Contribution**: Formal proof that LLMs cannot learn all computable functions, making hallucination inevitable for general-purpose use.
- **Relevance**: Establishes theoretical impossibility of eliminating hallucination — the seahorse emoji case is one instance of this fundamental limitation.

### 7. "Redefining 'Hallucination' in LLMs: Towards a Psychology-Informed Framework" (Berberette et al., 2024)
- **Authors**: Elijah Berberette, Jack Hutchins, Amir Sadovnik
- **Year**: 2024 (arXiv:2402.01769)
- **Key Contribution**: Proposes replacing "hallucination" with a taxonomy based on cognitive biases (confirmation bias, anchoring, source monitoring errors, etc.).
- **Relevance**: Provides theoretical framework for categorizing the seahorse emoji phenomenon as a specific type of confabulation rather than generic hallucination.

### 8. "Detecting Hallucinations in LLMs Using Semantic Entropy" (Farquhar et al., Nature 2024)
- **Authors**: Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, Yarin Gal
- **Year**: 2024 (Nature, 773+ citations)
- **Key Contribution**: Semantic entropy clusters generations by meaning before computing uncertainty, enabling confabulation detection.
- **Relevance**: Potential detection method for cases like the seahorse emoji — high semantic entropy across rephrased prompts would indicate unreliable outputs.

### 9. "Tokenization Matters! Degrading LLMs through Challenging Their Tokenization" (Wang et al., 2024)
- **Authors**: Dixuan Wang et al.
- **Year**: 2024 (arXiv:2405.17067)
- **Key Contribution**: Demonstrates LLM vulnerabilities from tokenization misalignment, especially with multi-byte UTF-8 characters.
- **Relevance**: Directly relevant — emoji tokenization creates "glitch token" effects where models cannot properly represent or validate emoji existence.

### 10. "A Comprehensive Survey of Hallucination in LLMs" (Zhang et al., 2025)
- **Year**: 2025 (arXiv:2510.06265)
- **Key Contribution**: Categorizes hallucination causes across 6 pipeline stages and detection into 5 technique families.
- **Relevance**: Provides taxonomic context for positioning our research.

---

## Common Methodologies

- **Logit lens / mechanistic interpretability**: Used to trace internal model representations (Vogel 2026, various interpretability works).
- **DRM false memory paradigm**: Adapted from cognitive psychology for LLMs (Cao et al. 2025).
- **Multi-agent interaction protocols**: For studying collective false memory (Xu et al. 2026).
- **Semantic entropy / uncertainty estimation**: For hallucination detection (Farquhar et al. 2024).
- **Human misinformation experiments**: For measuring downstream false memory impact (Chan et al. 2024).

## Standard Baselines

- **TruthfulQA**: Standard benchmark for testing whether models reproduce common falsehoods (817 questions).
- **Confabulations benchmark** (lechmazur): 201+ human-verified unanswerable questions for RAG evaluation.
- **MANBENCH**: 4,838 questions for collective false memory in multi-agent settings.

## Evaluation Metrics

- **False alarm rate / hit rate / d-prime**: From signal detection theory (Cao et al.).
- **Reality shift rate**: Proportion of correct→incorrect transitions under social influence (Xu et al.).
- **Semantic entropy**: Uncertainty in meaning space across generations (Farquhar et al.).
- **Confidence calibration (ECE)**: Expected Calibration Error (Kalai et al.).
- **False memory count and persistence**: Across immediate and delayed recall (Chan et al.).

## Datasets in the Literature

| Dataset | Papers Using It | Task |
|---------|----------------|------|
| DRM word association lists | Cao et al. 2025 | False memory formation in LLMs |
| TruthfulQA | Multiple hallucination studies | Factual accuracy and common misconceptions |
| BIG-Bench Hard (BBH) | Xu et al. 2026 (MANBENCH) | Knowledge-susceptible questions for Mandela effect |
| Unicode emoji data | Vogel 2026 (implicit) | Ground truth for emoji existence |
| Confabulations benchmark | lechmazur | RAG hallucination evaluation |

## Gaps and Opportunities

1. **No systematic study of "category completion" hallucinations**: The seahorse emoji case represents a broader pattern where LLMs infer existence of members within well-known categories (emoji sets, species lists, etc.) based on semantic similarity. This has not been systematically studied.

2. **Limited mechanistic understanding of negative knowledge**: How do LLMs represent "X does not exist"? The logit lens analysis shows models build representations for nonexistent things; understanding how to encode non-existence is an open question.

3. **No benchmark for "plausible nonexistent entities"**: Existing benchmarks test factual errors but not the specific case of entities that *should* exist based on category patterns but don't.

4. **Cross-model comparison gap**: The seahorse emoji phenomenon has been tested informally across models but lacks systematic experimental comparison with controlled conditions.

5. **Mitigation strategies untested for this class**: Cognitive anchoring and source scrutiny (from Mandela effect work) could be adapted but haven't been tested on category-completion hallucinations.

---

## Recommendations for Our Experiment

Based on the literature review:

- **Recommended datasets**: Unicode emoji list (ground truth), DRM word association lists (for cognitive analogue experiments), TruthfulQA (baseline comparison), custom emoji probe dataset (to be created).
- **Recommended baselines**: Multiple LLMs (GPT-4/5, Claude, Llama, Gemini) tested on emoji existence claims; TruthfulQA performance as calibration.
- **Recommended metrics**: False alarm rate for nonexistent emojis, confidence scores, semantic entropy across paraphrased queries, d-prime for emoji recognition.
- **Methodological considerations**:
  - Test both direct ("Does X emoji exist?") and indirect ("Show me the X emoji") probing.
  - Include "plausible nonexistent" items (seahorse, platypus, peacock) and "implausible nonexistent" items as controls.
  - Measure across temperature settings to assess confidence calibration.
  - Consider the tokenization pathway as a key mechanistic factor.
