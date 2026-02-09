# Generative Modeling via Drifting (Not specified in the paper.)
Source: Generative Modeling via Drifting.md

## Core reasons
- The paper proposes a new algorithmic paradigm where distribution evolution is performed during training, enabling one-step inference.
- The core contribution is a drifting-field-based computation/training mechanism (including equilibrium and fixed-point updates), not positional encoding, dimensional lifting, or benchmark construction.

## Evidence extracts
- "In this paper, we propose *Drifting Models*, a new paradigm for generative modeling. Drifting Models are characterized by learning a pushforward map that evolves during *training* time, thereby removing the need for an iterative inference procedure." (Section 1. Introduction)
- "To drive the evolution of the training-time pushforward, we introduce a *drifting field* that governs the sample movement." (Section 1. Introduction)
- "We present *Drifting Models*, a new paradigm for generative modeling. At the core of our model is the idea of modeling the evolution of pushforward distributions *during training*. This allows us to focus on the update rule, *i.e.*,  $\\mathbf{x}_{i+1} = \\mathbf{x}_i + \\Delta \\mathbf{x}_i$ , during the iterative training process. This is in contrast with diffusion-/flow-based models, which perform the iterative update at *inference* time. Our method naturally performs one-step inference." (Section 6. Discussion and Conclusion)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
