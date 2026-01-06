# Context-aware Rotary Position Embedding (Not specified in the paper.)
Source: Context-aware Rotary Position Embedding (CARoPE).md

## Core reasons
- The paper critiques RoPE's static sinusoidal frequencies and frames them as limiting context-sensitive positional modeling within Transformers.
- The central contribution is a new positional encoding mechanism (CARoPE) that dynamically generates input- and head-dependent rotary frequencies.

## Evidence extracts
- "However, RoPE relies on static, input-independent sinusoidal frequency patterns, limiting its ability to model context-sensitive relationships." (Abstract)
- "we propose CARoPE (Context-Aware Rotary Positional Embedding), a novel enhancement of RoPE that introduces dynamic, input-dependent frequency values for each attention head." (1 Introduction)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
