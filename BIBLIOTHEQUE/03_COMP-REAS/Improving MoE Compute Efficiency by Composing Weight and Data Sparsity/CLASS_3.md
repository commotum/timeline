# Improving MoE Compute Efficiency by Composing Weight and Data Sparsity (Not specified in the paper)
Source: Improving MoE Compute Efficiency by Composing Weight and Data Sparsity.md

## Core reasons
- Proposes a routing mechanism that changes how computation is allocated per token by composing weight and data sparsity with null experts.
- Focuses on variable compute within MoE layers (a computation mechanism), not positional encoding, dimensional lifting, or datasets/benchmarks.

## Evidence extracts
- "We recover data sparsity within causal token-choice MoE by leveraging zero-compute (null) experts within the routing pool." (Abstract)
- "We extend token-choice MoE with a minimal modification: adding null experts to the routing pool. This composes weight and data sparsity while preserving causality." (Section 4 METHOD)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
