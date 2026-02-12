# DiffSAT: Differential MaxSAT Layer for SAT Solving (Year not specified)
Source: DiffSAT- Differential MaxSAT Layer for SAT Solving.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract presents DiffSAT as a differential MaxSAT solver layer with iterative forward/backward optimization, not a Transformer/self-attention architecture.
- The auxiliary files characterize the method as SAT assignment search with a parameter-free MaxSAT layer; no central self-attention block is indicated for DiffSAT.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files were consistent and sufficient.

## Evidence
- "we introduce DiffSAT, a novel approach that differentiates the discrete SAT problem and progressively searches for satisfying assignments through the forward and backward propagation of a neural network layer." (Abstract, `DiffSAT- Differential MaxSAT Layer for SAT Solving.md`)
- "DiffSAT aims to determine the satisfying assignment for a given CNF formula." (Section 4.1 quote captured in `TASK-DOMAINS.md`)
- "DiffSAT does not have any parameters and therefore does not require any training labels." (Section 4.1 quote captured in `TASK_MODEL_RATIO.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Read abstract + `TASK-DOMAINS.md` + `TASK-DOMAINS.csv` + `TASK_MODEL_RATIO.md` in full; extending-dimensions file was unavailable (`MISSING`); evidence was sufficient for high-confidence classification.
Pass 2 (targeted source scan): skipped - Pass 1 already provided sufficient evidence.
