# HoPE: Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling in Large Language Models (Not specified in the paper.)
Source: HoPE- Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling in Large Language Models.md

## Core reasons
- The paper critiques existing positional encodings (absolute, ALiBi, RoPE) and identifies instability or degradation in long-range contexts.
- The main contribution is a new positional encoding method (HoPE) that modifies rotary positional encoding using hyperbolic rotations to improve long-range attention behavior.

## Evidence extracts
- "While absolute positional encodings struggle with extrapolation to longer sequences due to fixed positional representations, and relative approaches like Alibi exhibit performance degradation on extremely long contexts, the widely-used Rotary Positional Encoding (RoPE) introduces oscillatory attention patterns that hinder stable longdistance dependency modelling." (Abstract)
- "we propose Hyperbolic Rotary Positional Encoding (HoPE), which leverages hyperbolic functions to implement Lorentz rotations on token representations." (Abstract)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
