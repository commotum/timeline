# GPT-4 Technical Report (2023)
Source: GPT-4 Technical Report.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract directly identifies GPT-4 as a Transformer-based model used for the paper’s main results.
- Auxiliary analyses characterize GPT-4 as a Transformer-style model across tasks, indicating self-attention architecture is central rather than peripheral.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the available Pass 1 evidence is sufficient for a high-confidence decision.

## Evidence
- "GPT-4 is a Transformer-based model pre-trained to predict the next token in a document." (GPT-4 Technical Report.md, Abstract)
- "GPT-4 is a Transformer-style model [39] pre-trained to predict the next token in a document, using both publicly available data (such as internet data) and data licensed from third-party providers." (TASK_MODEL_RATIO.md, quoted from GPT-4 Technical Report.md Section 2)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - high-confidence TRANSFORMER-YES from explicit architecture statements in abstract and auxiliary files; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient.
