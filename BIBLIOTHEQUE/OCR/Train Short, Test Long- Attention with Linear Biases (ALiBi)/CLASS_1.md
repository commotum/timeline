# TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION (Not specified in the paper.)
Source: Train Short, Test Long- Attention with Linear Biases (ALiBi).md

## Core reasons
- The paper identifies limitations in existing positional embedding methods for extrapolation and attributes the issue to the position embedding method itself.
- The main contribution is a new positional method (ALiBi) that changes how position is handled by biasing attention scores instead of adding positional embeddings.

## Evidence extracts
- "We demonstrate that this failure to extrapolate is caused by the position embedding method." (Section 1 Introduction)
- "We therefore introduce a simpler and more efficient position method, Attention with Linear Biases (ALiBi). ALiBi does not add positional embeddings to word embeddings; instead, it biases query-key attention scores with a penalty that is proportional to their distance." (Abstract)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
