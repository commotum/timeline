# Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free (Not specified in the paper)
Source: Gated Attention for LLMs- Non-linearity, Sparsity, Sink-Free.md

## Core reasons
- Proposes adding gating to the softmax attention computation to introduce non-linearity and sparse, query-dependent modulation of SDPA outputs.
- Focuses on an architectural computation change within attention (gating positions and mechanisms) rather than datasets, positional encoding, or dimensional lifting.

## Evidence extracts
- "By comparing various gating positions and computational variants, we attribute this effectiveness to two key factors: (1) introducing non-linearity upon the low-rank mapping in the softmax attention, and (2) applying query-dependent sparse gating scores to modulate the SDPA output." (Abstract)
- "In this work, we investigate gating mechanisms in the standard softmax attention (Vaswani, 2017) (Sec.2.2). Specifically, we introduce gating at distinct positions (Fig. 1): after the query  $(G_4)$ , key  $(G_3)$ , and value projections  $(G_2)$ ; following the Scaled Dot Product Attention (SDPA) outputs  $(G_1)$ ; and after the final dense output layer  $(G_5)$ ." (Section 1 Introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
