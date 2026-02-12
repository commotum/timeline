# Language Models are Unsupervised Multitask Learners (2019)
Source: Language Models are Unsupervised Multitask Learners.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly identifies the central model as GPT-2 and states it is a Transformer, so Transformer-style self-attention is core to the paper’s main model and results.
- Auxiliary files are consistent with GPT-2 as the single model across tasks; the extending-dimensions analysis file was unavailable (`MISSING`) and was skipped.

## Evidence
- "Our largest model, GPT-2, is a 1.5B parameter Transformer that achieves state of the art results on 7 out of 8 tested language modeling datasets in a zero-shot setting but still underfits WebText." (Abstract, `Language Models are Unsupervised Multitask Learners.md`)
- "2. **Number of trained model instances required to cover all tasks:** 1 model" (`TASK_MODEL_RATIO.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence TRANSFORMER-YES from the abstract and auxiliary analyses; extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - not needed because Pass 1 already contains explicit Transformer architecture evidence.
