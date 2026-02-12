# Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes (Year not specified)
Source: Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: source-targeted-scan

## Why
- The central model is Sparse Access Memory (SAM), a memory-augmented neural network built around external memory read/write operations rather than Transformer blocks.
- The architecture uses an LSTM controller and NTM/DNC-style addressing; self-attention as the core computation is not presented.

## Evidence
- "we present an end-to-end differentiable memory access scheme, which we call Sparse Access Memory (SAM)" (Abstract, `Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes.md`)
- "We use a one layer LSTM for the controller throughout." (Section 3.3 Controller, `Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Abstract, `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md` reviewed; extending-dimensions analysis file was unavailable (`MISSING`).
Pass 2 (targeted source scan): performed - Checked architecture sections in the paper markdown to confirm LSTM/NTM-style memory access and absence of Transformer-style self-attention as the core model.
