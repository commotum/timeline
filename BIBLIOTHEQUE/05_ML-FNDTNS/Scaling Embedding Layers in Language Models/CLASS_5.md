# Scaling Embedding Layers in Language Models (Not specified in the paper.)
Source: Scaling Embedding Layers in Language Models.md

## Core reasons
- The paper’s main contribution is a new method (SCONE) for scaling input embedding layers in language models, focusing on training and inference efficiency rather than positional encoding or higher-dimensional lifting.
- It proposes scaling strategies (more f-gram embeddings and larger f-gram models) that improve performance while keeping inference-time accelerator usage fixed, which fits ML foundations/training-method contributions.

## Evidence extracts
- "We propose SCONE (Scalable, Contextualized, Offloaded, N-gram Embedding), a new method for extending input embedding layers to enhance language model performance." (Abstract)
- "SCONE introduces two novel scaling approaches for improving model performance: (i) increasing the number of cached f-gram embeddings and (ii) scaling up the f-gram model used to learn these embeddings." (1 Introduction)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
