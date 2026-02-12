# Predictability and Surprise in Large Generative Models (2022)
Source: Predictability and Surprise in Large Generative Models.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract centers the paper on GPT-3, Megatron-Turing NLG, and Gopher, which are Transformer-family large language models.
- Auxiliary analyses repeatedly frame the evaluated systems as GPT-3/Gopher/"GPT-3-like language models," indicating the main results are about LLM behavior rather than non-attention architectures.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files are sufficient for a high-confidence architecture-family decision.

## Evidence
- "Large-scale pre-training has recently emerged as a technique for creating capable, general-purpose, generative models such as GPT-3, Megatron-Turing NLG, Gopher, and many others." (Abstract text block, `Predictability and Surprise in Large Generative Models.md`)
- "In this paper, we attempt to better understand the influence of scaling laws on the dynamics of large-scale model development and deployment, with a focus on large language models." (Introduction, `Predictability and Surprise in Large Generative Models.md`)
- "Fig. 2 ... based on three different models: GPT-3 (blue), Gopher (orange), and a Google language model (green)." (Section 2.2 quote recorded in `TASK_MODEL_RATIO.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence TRANSFORMER-YES decision using abstract + `TASK-DOMAINS.md` + `TASK-DOMAINS.csv` + `TASK_MODEL_RATIO.md`; extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient for final classification.
