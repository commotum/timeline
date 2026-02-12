# Toolformer: Language Models Can Teach Themselves to Use Tools (Year not specified)
Source: Toolformer- Language Models Can Teach Themselves to Use Tools.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The central model is Toolformer built on GPT-J across experiments; GPT-J is a GPT-family Transformer language model using self-attention blocks.
- The abstract frames Toolformer as a language-model method, and the auxiliary model-ratio analysis repeatedly identifies GPT-J as the underlying model instance.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files were sufficient for a high-confidence decision.

## Evidence
- "Language models (LMs) exhibit remarkable abilities ... We introduce Toolformer, a model trained to decide which APIs to call ..." (Toolformer- Language Models Can Teach Themselves to Use Tools.md, Abstract, line 9)
- "Throughout all of our experiments, we use a subset of CCNet (Wenzek et al., 2020) as our language modeling dataset C and GPT-J (Wang and Komatsuzaki, 2021) as our language model M." (TASK_MODEL_RATIO.md, Section 4.1 quote, line 17)
- "Toolformer: GPT-J finetuned on  C^* , our subset of CCNet augmented with API calls." (TASK_MODEL_RATIO.md, Section 4.1 quote, line 19)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence from abstract + TASK-DOMAINS/TASK-DOMAINS.csv/TASK_MODEL_RATIO; extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided high-confidence GPT-family Transformer identification.
