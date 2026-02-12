# LONG SHORT-TERM MEMORY (1997)
Source: Long Short-Term Memory.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract defines the central method as Long Short-Term Memory (LSTM), a recurrent gradient-based architecture with gating, not a Transformer self-attention architecture.
- The auxiliary analyses describe LSTM cell-block sequence models and do not identify Transformer-family blocks (BERT/GPT/ViT-style attention) as part of the core method.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files were sufficient for a high-confidence binary decision.

## Evidence
- "introducing a novel, efficient, gradient-based method called \"Long Short-Term Memory\" (LSTM)." (Abstract, `Long Short-Term Memory.md:21`)
- "Multiplicative gate units learn to open and close access to the constant error flow." (Abstract, `Long Short-Term Memory.md:21`)
- "Architecture. We use a 3-layer net with 2 input units, 1 output unit, and 2 cell blocks of size 2." (Section 5.4 citation, `TASK_MODEL_RATIO.md:43`)
- "Across tasks, the setups imply static attention over the full input stream and constructed internal state to retain long-range information." (`TASK-DOMAINS.md:20`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence TRANSFORMER-NO from abstract plus available auxiliary files.
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient; additional full-paper scanning was not required.
