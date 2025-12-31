# CLOSED-LOOP TRANSFORMERS: AUTOREGRESSIVE MODELING AS ITERATIVE LATENT EQUILIBRIUM (2025)
Source: be1734-2025.pdf

## Core reasons
- The paper argues standard autoregressive transformers have an open-loop limitation with uncorrected error propagation, indicating a missing capability for iterative correction.
- It proposes a closed-loop mechanism where latent states are iteratively refined to a self-consistent equilibrium before token emission, changing the computation process.

## Evidence extracts
- "Contemporary autoregressive transformers operate in open loop: each hidden state is computed in a single forward pass and never revised, causing errors to propagate uncorrected through the sequence." (Abstract)
- "We introduce the closed-loop prediction principle, which requires that before emitting any token, the model must iteratively refine its latent representation until it reaches a self-consistent equilibrium with respect to an internal energy function." (Introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
