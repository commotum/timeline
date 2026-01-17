# Rotary Masked Autoencoders are Versatile Learners (Not specified in the paper.)
Source: Rotary Masked Autoencoders Are Versatile Learners.md

## Core reasons
- The paper's main contribution includes using Rotary Positional Embeddings for continuous positions to handle irregular sampling, which is a positional encoding change rather than a new model family.
- It explicitly identifies the limitation of standard Transformer positional information for irregular time-series and frames RoMAE as addressing that positional limitation.

## Evidence extracts
- "We present the Rotary Masked Autoencoder (RoMAE), which utilizes the popular Rotary Positional Embedding (RoPE) method for continuous positions." (Abstract)
- "Being originally designed for sequences of text, the base Transformer architecture is not able to deal with such irregularly sampled data, by default only supporting quantized positional information as is found in natural language." (Section 1 Introduction)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
