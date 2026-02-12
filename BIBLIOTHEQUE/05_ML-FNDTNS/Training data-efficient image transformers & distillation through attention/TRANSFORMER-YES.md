# Training data-efficient image transformers & distillation through attention (2021)
Source: Training data-efficient image transformers & distillation through attention.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract directly identifies the method as vision transformers and attention-based models, indicating self-attention is central to the paper’s main approach.
- The auxiliary analyses consistently describe transformer blocks, patch-token self-attention, and distillation-through-attention as part of the core method and reported results.

## Evidence
- "Recently, neural networks purely based on attention were shown to address image understanding tasks such as image classification." (Training data-efficient image transformers & distillation through attention.md, Abstract)
- "In this work, we produce competitive convolutionfree transformers trained on ImageNet only using a single computer in less than 3 days." (Training data-efficient image transformers & distillation through attention.md, Abstract)
- "We show that our neural networks that contain no convolutional layer can achieve competitive results against the state of the art on ImageNet with no external data." (TASK_MODEL_RATIO.md, item 1)
- "This transformer-specific strategy outperforms vanilla distillation by a significant margin." (TASK-DOMAINS.md, Evidence section)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence TRANSFORMER-YES from the abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md; Extending-dimensions analysis markdown was unavailable (MISSING).
Pass 2 (targeted source scan): skipped - Pass 1 already provided clear, direct transformer/self-attention evidence.
