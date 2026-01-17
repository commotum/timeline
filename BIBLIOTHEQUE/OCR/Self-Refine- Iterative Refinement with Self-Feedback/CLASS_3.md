# SELF-REFINE: Iterative Refinement with Self-Feedback (Not specified in the paper.)
Source: Self-Refine- Iterative Refinement with Self-Feedback.md

## Core reasons
- Proposes an iterative self-feedback/refinement loop that changes how generation is computed at test time.
- Centers the contribution on a new computation mechanism (feedback then refine with the same LLM), not on data or positional encoding.

## Evidence extracts
- "the same LLM provides *feedback* for its output and uses it to *refine* itself, iteratively." (Abstract)
- "Given an input sequence, SELF-REFINE generates an initial output, provides feedback on the output, and refines the output according to the feedback." (Section 2 Iterative Refinement with SELF-REFINE)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
