# Cloned Reference Repositories

Code repositories collected for the research project: "Understanding False Memories in Large Language Models: The Case of the Nonexistent Seahorse Emoji"

## 1. ai-false-memories

- **Source**: https://github.com/mitmedialab/ai-false-memories
- **Paper**: Chan et al. (2024) "Conversational AI Powered by Large Language Models Amplifies Human False Memories"
- **Relevance**: Studies how LLM-based chatbots amplify false memory formation in humans through suggestive questioning. Provides experimental framework (control vs survey vs pre-scripted chatbot vs generative chatbot) and data analysis pipelines for false memory measurement.
- **Key contents**:
  - `Data/Code/` - Jupyter notebooks for false memory analysis (immediate and 1-week follow-up)
  - `Data/Raw/`, `Data/Processed/` - De-identified experimental data
  - `Prototype/` - Implementations of survey, pre-scripted chatbot, and generative chatbot conditions
  - `Supplementary/` - Survey materials and crime video used in experiments

## 2. LLM_CognitivePsychology

- **Source**: https://github.com/zycao29/LLM_CognitivePsychology.git
- **Paper**: Cao, Schooler & Zafarani (2025) "Analyzing Memory Effects in Large Language Models through the lens of Cognitive Psychology" (arXiv:2509.17138)
- **Relevance**: Directly applies cognitive psychology memory paradigms (including DRM - Deese-Roediger-McDermott) to LLMs. Tests whether LLMs exhibit human-like false memory effects.
- **Key contents**:
  - `offline_LM/` - Jupyter notebooks for local models (Llama3, Mistral, Gemma2, Phi-4, Qwen2, Falcon, SOLAR, InternLM, Zephyr)
  - `online_LM(GPT)/` - Scripts for GPT API-based experiments (`llm_DRM.py`, `llm_memory.py`)
  - `prompt.txt` - Prompt templates used in experiments
- **Dependencies**: Transformers, OpenAI API key for online experiments

## 3. confabulations

- **Source**: https://github.com/lechmazur/confabulations
- **Paper/Benchmark**: LLM Confabulation (Hallucination) Leaderboard for RAG
- **Relevance**: Comprehensive benchmark for measuring LLM confabulation/hallucination rates using misleading questions based on provided text documents. Useful for comparing how different models handle questions about non-existent information (analogous to the seahorse emoji problem).
- **Key contents**:
  - `questions/` - 201+ curated questions designed to elicit confabulations
  - `prompts/` - Prompt templates (initial, compound, reminder variants)
  - `responses/` - Model responses from 60+ models
  - `articles/` - Source text documents
  - `leaderboard1.html` - Interactive leaderboard
- **Notable findings**: Claude models tend toward lower confabulation but higher non-response rates; reasoning models show varying trade-offs

## 4. Mandela-Effect

- **Source**: https://github.com/bluedream02/Mandela-Effect
- **Paper**: Xu et al. (2026) "When Agents 'Misremember' Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems" (ICLR 2026, arXiv:2602.00428)
- **Relevance**: Directly studies collective false memories (Mandela effect) in LLM multi-agent systems. Introduces ManBench benchmark with 4,838 questions across 20 tasks. Tests 13 LLMs with 5 interaction protocols and evaluates defense strategies.
- **Key contents**:
  - `eval.py` - Main evaluation script for running ManBench
  - `eval_defense.py` - Evaluation with defense strategies
  - `eval_correct.py` - Correctness evaluation
  - `analyze.py` - Result analysis and reporting
  - `data/bbh_all_small/` - Benchmark task data
  - `utils.py` - Utility functions
- **Dependencies**: OpenAI API key, Python 3.10, conda environment recommended

## 5. seahorse-emoji

- **Source**: https://gist.github.com/vgel/025ad6af9ac7f3bc194966b03ea68606
- **Author**: Theia (vgel.me) - "Why do LLMs freak out over the seahorse emoji?"
- **Relevance**: Core to our research question. Demonstrates via logit lens interpretability that LLMs build an internal concept of "seahorse + emoji" in middle layers, but the output layer snaps to the nearest real token (fish/horse emoji) since no seahorse emoji exists. The seahorse emoji was proposed to Unicode but rejected in 2018.
- **Key contents**:
  - `seahorse_test.py` - Script using Llama 3.3 70B with logit lens to analyze token-level false belief
- **Dependencies**: transformers, accelerate, 2xH200 GPUs (for 70B model)

---

## Research Connections

| Repository | False Memory Type | Subject | Method |
|---|---|---|---|
| ai-false-memories | Human false memories amplified by AI | Humans interacting with LLMs | Behavioral experiment (N=200) |
| LLM_CognitivePsychology | DRM-style false memories in LLMs | LLMs as subjects | Cognitive psychology paradigms |
| confabulations | Confabulation/hallucination | LLMs as subjects | RAG-based misleading questions |
| Mandela-Effect | Collective false memories in multi-agent LLMs | Multi-agent LLM systems | ManBench benchmark (4,838 questions) |
| seahorse-emoji | Nonexistent concept false belief | LLMs as subjects | Logit lens interpretability |
