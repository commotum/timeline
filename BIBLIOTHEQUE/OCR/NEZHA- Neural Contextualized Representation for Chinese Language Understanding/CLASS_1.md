# NEZHA: Neural Contextualized Representation for Chinese Language Understanding (2021)
Source: NEZHA- Neural Contextualized Representation for Chinese Language Understanding.md

## Core reasons
- The paper's main contribution is a transformer-based model that introduces a new positional encoding scheme, namely functional relative positional encoding.
- It identifies the need for positional encoding because standard self-attention is permutation invariant and then replaces prior absolute/parametric encodings with a relative, function-based approach.

## Evidence extracts
- "The current version of NEZHA is based on BERT [1] with a collection of proven improvements, which include Functional Relative Positional Encoding as an effective positional encoding scheme, Whole Word Masking strategy, Mixed Precision Training and the LAMB Optimizer in training the models." (Abstract)
- "Since the multi-head attention in Transformer (and BERT) is permutation invariant, and thus not sensitive to the word order. Therefore, [9] incorporates an absolute positional encoding for each position, which is an embedding vector and added to the token embedding directly." (Section 2.1 Preliminaries: BERT Model & Positional Encoding)
- "In this technical report, we employ a functional relative positional encoding scheme, which encodes the relative positions in self-attention by pre-defined functions without any trainable parameter." (Section 1 Introduction)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
