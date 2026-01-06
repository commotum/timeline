# Base of RoPE Bounds Context Length (Not specified in the paper.)
Source: Base of RoPE Bounds Context Length.md

## Core reasons
- The paper centers on RoPE positional embedding, critiques existing long-context extrapolation based on OOD rotation angles, and derives a base-dependent bound to guide positional encoding behavior.
- The core contribution is a change in how positional encoding is configured (deriving an absolute lower bound for RoPE base to achieve a target context length), not a new dataset or architecture.

## Evidence extracts
- "We revisit the role of RoPE in LLMs and propose a novel property of long-term decay, we derive that the *base of RoPE bounds context length*: there is an absolute lower bound for the base value to obtain certain context length capability." (Abstract)
- "In summary, the RoPE's base determines the upper bound of context length the model can truly obtain." (Section 4.3)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
