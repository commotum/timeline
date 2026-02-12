# Less is More: Recursive Reasoning with Tiny Networks (Year not specified)
Source: Less is More- Recursive Reasoning with Tiny Networks.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: medium
Basis: abstract-aux-only

## Why
- The auxiliary analysis states that the reported TRM results use a self-attention variant on multiple core benchmarks (Maze-Hard, ARC-AGI-1, ARC-AGI-2), so self-attention is materially used in main results.
- The paper also includes an attention-free variant, but self-attention remains part of the central TRM model family rather than only a peripheral comparison.
- The Extending-dimensions analysis file was unavailable (`MISSING`), so the decision is based on the abstract and available auxiliary files.

## Evidence
- "From the results, we see that TRM without selfattention obtains the best generalization on Sudoku-Extreme (87.4% test accuracy). Meanwhile, TRM with self-attention generalizes better on the other tasks..." (TASK_MODEL_RATIO.md, Section 5 quote)
- "TRM with self-attention obtains 85.3% accuracy on Maze-Hard, 44.6% accuracy on ARC-AGI-1, and 7.8% accuracy on ARC-AGI-2 with 7M parameters." (TASK_MODEL_RATIO.md, Section 5 quote)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a confident decision from abstract + TASK-DOMAINS.md + TASK-DOMAINS.csv + TASK_MODEL_RATIO.md; Extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided explicit self-attention usage in core results.
