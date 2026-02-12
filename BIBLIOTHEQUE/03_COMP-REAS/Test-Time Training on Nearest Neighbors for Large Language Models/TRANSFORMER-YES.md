# Test-Time Training on Nearest Neighbors for Large Language Models (Year not specified)
Source: Test-Time Training on Nearest Neighbors for Large Language Models.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly frames the core method around adapting "modern Transformers" at test time, indicating Transformer architecture is central rather than peripheral.
- The reported main results are on GPT-2 and GPT-Neo language models, which are Transformer-family models used as the primary systems in the study.
- The Extending-dimensions analysis markdown was unavailable (`MISSING`), but the abstract plus available auxiliary files already provide direct architecture evidence.

## Evidence
- "...cost in computation and memory grows quadratically for modern Transformers. To avoid these complications, we simply fine-tune the model on retrieved data at test time..." (Abstract, `Test-Time Training on Nearest Neighbors for Large Language Models.md`)
- "...test-time training with nearest neighbors significantly narrows the performance gap between a small GPT-2 and a GPT-Neo model more than 10 times larger." (Abstract, `Test-Time Training on Nearest Neighbors for Large Language Models.md`)
- "Figure 5: Bits per byte results on all Pile tasks for a small GPT-2 model (117M parameters) before and after test-time training on 50 nearest neighbors." (`TASK_MODEL_RATIO.md`, item 2)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence Transformer-family classification from abstract and auxiliary analyses.
Pass 2 (targeted source scan): skipped - Not needed after clear Pass 1 evidence.
