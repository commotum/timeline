# Resonance RoPE: Improving Context Length Generalization of Large Language Models (Not specified in the paper.)
Source: Resonance RoPE- Improving Context Length Generalization of Large Language Models.md

## Core reasons
- The paper's main contribution is a modification to RoPE positional embeddings (RESONANCE ROPE) to reduce generalization gaps on out-of-distribution positions.
- It critiques limitations of existing RoPE scaling approaches and proposes changing the interpolation behavior of RoPE features rather than altering model dimensions or tasks.

## Evidence extracts
- "We introduce RESONANCE ROPE, a novel approach designed to narrow the generalization gap in TSTL scenarios by refining the interpolation of RoPE features for OOD positions" (Abstract)
- "We tackle this issue by developing a synergistic modification to the conventional RoPE embedding, referred to as RESONANCE ROPE." (Section 4 Proposed Method: RESONANCE ROPE)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
