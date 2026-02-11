# AI Feynman: a Physics-Inspired Method for Symbolic Regression (2020)
Source: AI Feynman- a Physics-Inspired Method for Symbolic Regression.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: hint-only

## Why
- Hint files describe the method as symbolic regression with brute-force symbolic expression search and neural-network-based helper components, not Transformer blocks.
- The hints explicitly mark attention dynamics as not specified, with no indication of self-attention being central to the model used for results.

## Evidence
- "Attention Dynamic | Not specified in the paper." (TASK-DOMAINS.md, Task Table)
- "In order to obtain such an interpolating function for a given mystery, we train a neural network to predict the output given its input." (TASK_MODEL_RATIO.md, Section II.E.1 quote)

## Pass accounting
Pass 0 (hint-first): performed - Hints gave sufficient evidence that the central method is symbolic regression + neural network support, with no Transformer/self-attention model indicated.
Pass 1 (source triage): skipped - High-confidence decision from hint files.
Pass 2 (source deep dive): skipped - Not needed after hint-only resolution.
