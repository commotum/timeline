# AI Feynman: A physics-inspired method for symbolic regression (2020)
Source: AI Feynman.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: hint-only

## Why
- The task and method are framed as symbolic regression from tabular numeric data to analytic expressions, not sequence modeling with self-attention.
- Hint files describe training a neural network interpolator per mystery and explicitly report attention dynamics as "Not specified in the paper," with no transformer-style blocks indicated.

## Evidence
- "our task is to discover the correct symbolic expression for the unknown mystery function f, optionally including the complication of noise." (TASK-DOMAINS.md, INTRODUCTION)
- "To obtain such an interpolating function for a given mystery, we train a neural network" (TASK_MODEL_RATIO.md, Section: Neural network training)

## Pass accounting
Pass 0 (hint-first): performed - sufficient evidence for high-confidence TRANSFORMER-NO from TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md.
Pass 1 (source triage): skipped - hint evidence already decisive; no transformer/self-attention cues in hints.
Pass 2 (source deep dive): skipped - not needed after decisive hint-first triage.
