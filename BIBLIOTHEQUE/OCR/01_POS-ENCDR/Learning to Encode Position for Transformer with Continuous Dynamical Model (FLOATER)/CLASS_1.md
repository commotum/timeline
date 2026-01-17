# Learning to Encode Position for Transformer with Continuous Dynamical Model (Not specified in the paper.)
Source: Learning to Encode Position for Transformer with Continuous Dynamical Model (FLOATER).md

## Core reasons
- The paper critiques limitations of existing positional encodings (fixed sinusoidal and learned embeddings) in Transformers and argues for a learnable, inductive alternative.
- It proposes FLOATER, a new position encoder that models position information with a continuous dynamical system.

## Evidence extracts
- "However, this solution has clear limitations: the sinusoidal encoding is not flexible enough as it is manually designed and does not contain any learnable parameters, whereas the position embedding restricts the maximum length of input sequences." (Abstract)
- "- We propose FLOATER, a new position encoder for Transformer, which models the position information via a continuous dynamical model in a data-driven and parameter-efficient manner." (Section 1 Introduction)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
