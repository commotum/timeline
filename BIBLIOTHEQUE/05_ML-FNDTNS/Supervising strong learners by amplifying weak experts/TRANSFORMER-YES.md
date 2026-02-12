# Supervising strong learners by amplifying weak experts (Year not specified)
Source: Supervising strong learners by amplifying weak experts.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The paper’s core learner `X` is explicitly implemented as an encoder-decoder with self-attention and described as closely following the Transformer architecture.
- The auxiliary analyses align with Transformer-centric model cues, and no alternative non-attention central architecture is indicated; the extending-dimensions file was unavailable (`MISSING`).

## Evidence
- "We implement X as an encoder-decoder architecture with self-attention, closely following the Transformer architecture (Vaswani et al., 2017):" (Supervising strong learners by amplifying weak experts.md, Section 2.5 "Model architecture", line 108)
- "The human-predictor H' is also a Transformer decoder augmented with the ability to copy symbols from previous steps." (Supervising strong learners by amplifying weak experts.md, Section 2.5 "Model architecture", line 116)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Reviewed abstract, `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; extending-dimensions analysis was unavailable (`MISSING`), and abstract-level evidence alone did not fully establish architecture centrality.
Pass 2 (targeted source scan): performed - Scanned the model-architecture section and found explicit self-attention/Transformer implementation for the central model.
