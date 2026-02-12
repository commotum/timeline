# Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models (2024)
Source: Language Agent Tree Search (LATS).md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract makes large language models the core engine of the method and reports main results specifically with GPT-family models (GPT-4, GPT-3.5), which are Transformer-family architectures.
- Auxiliary files reinforce that one LM-based framework is used across all evaluated tasks; this is sufficient to classify as Transformer-based even though the extending-dimensions file was unavailable (`MISSING`).

## Evidence
- "By leveraging the in-context learning ability of LMs, we integrate Monte Carlo Tree Search into LATS to enable LMs as agents, along with LM-powered value functions and self-reflections for proficient exploration and enhanced decision-making." (Abstract, `Language Agent Tree Search (LATS).md`)
- "Notably, LATS achieves state-of-the-art pass@1 accuracy (92.7%) for programming on HumanEval with GPT-4 and demonstrates gradient-free performance (average score of 75.9) comparable to gradient-based fine-tuning for web navigation on WebShop with GPT-3.5." (Abstract, `Language Agent Tree Search (LATS).md`)
- "As LATS does not involve training, we propose a novel value function for this setting based on two components: (1) a self-generated LM score and (2) a self-consistency score." (Section 4.2 quote captured in `TASK_MODEL_RATIO.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for high-confidence decision from abstract + `TASK-DOMAINS.md` + `TASK-DOMAINS.csv` + `TASK_MODEL_RATIO.md`; extending-dimensions analysis file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient to finalize.
