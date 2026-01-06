# Improving language models by retrieving from trillions of tokens (Not specified in the paper.)
Source: RETRO-style retrieval-augmented pretraining and variants.md

## Core reasons
- Proposes a retrieval-augmented language modeling mechanism that conditions predictions on retrieved chunks from a large external database, adding explicit memory to inference.
- Introduces a chunked cross-attention and encoder-decoder integration of retrieved neighbors, changing the computation pathway rather than positional encoding or dimensionality.

## Evidence extracts
- "We enhance auto-regressive language models by conditioning on document chunks retrieved from a large corpus, based on local similarity with preceding tokens." (Abstract)
- "Our model relies on an encoder-decoder transformer architecture, integrating the retrieved data through a cross-attention mechanism as introduced in Vaswani et al. (2017)." (Section 2.4)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
