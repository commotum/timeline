# End-to-End Differentiable Proving (Year not specified)
Source: End-to-End Differentiable Proving.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes Neural Theorem Provers built from Prolog-style backward chaining and differentiable unification with an RBF kernel, not Transformer blocks or self-attention.
- The auxiliary task/model analyses do not provide any concrete Transformer-family cue; the only "attention" mention is marked as inferred task dynamics rather than architectural self-attention.

## Evidence
- "Specifically, we replace symbolic unification with a differentiable computation on vector representations of symbols using a radial basis function kernel" (Abstract, `End-to-End Differentiable Proving.md`)
- "The reasoning uses bounded proof depth and predefined rule templates ... (dynamic attention/state, inferred)." (`TASK-DOMAINS.md`, Summary)
- "Attention Dynamic | Dynamic (inferred)" (`TASK-DOMAINS.md`, Task Table)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence NON-Transformer classification; no central self-attention/Transformer architecture indicated, and extending-dimensions file was unavailable (MISSING).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient for a high-confidence decision.
