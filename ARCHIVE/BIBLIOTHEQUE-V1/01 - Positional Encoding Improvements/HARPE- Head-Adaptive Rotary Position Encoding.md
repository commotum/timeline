# Breaking the Stage Barrier: A Novel Single-Stage Approach to Long Context Extension for Large Language Models (2025)
Source: c37494-2025.pdf

## Core reasons
- The paper critiques multi-stage long-context positional encoding pipelines for requiring extensive manual tuning, motivating a simpler single-stage alternative.
- The core contribution is Head-Adaptive Rotary Position Encoding (HARPE), which changes RoPE base frequencies across attention heads to extend context length during training.

## Evidence extracts
- "However, those approaches require extensive manual tuning and human expertise." (Abstract)
- "Our HARPE leverages different Rotary Position Encoding (RoPE) base frequency values across different attention heads and directly trains LLMs on the target context length." (Abstract)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
