# Scaling Up Vision-Language Learning With Noisy Text Supervision (ALIGN) (2021)
Source: Scaling Up Vision-Language Learning With Noisy Text Supervision (ALIGN).md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The central ALIGN architecture explicitly uses BERT as its text encoder; BERT is a Transformer family model with self-attention.
- The BERT text tower is part of the core dual-encoder used for the paper’s main pre-training and transfer results, not a peripheral baseline.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files and targeted architecture lines were sufficient.

## Evidence
- "A simple dual-encoder architecture learns to align visual and language representations of the image and text pairs using a contrastive loss." (Abstract, `Scaling Up Vision-Language Learning With Noisy Text Supervision (ALIGN).md`)
- "We use EfficientNet with global pooling (without training the 1x1 conv layer in the classification head) as the image encoder and BERT with [CLS] token embedding as the text embedding encoder (we generate 100k wordpiece vocabulary from our training dataset)." (Section 4.1, `Scaling Up Vision-Language Learning With Noisy Text Supervision (ALIGN).md`)
- "Inference: In Dimension/In Dynamics, Attention Dynamic, State Dynamic, Out Dimension, and Out Dynamics are inferred from \"The model consists of a pair of image and text encoders with a cosine-similarity combination function at the top.\" (Section 4.1) and \"For BERT we use wordpiece sequence of maximum 64 tokens\" plus \"The image encoder is trained at resolution of  $289 \times 289$  pixels\" (Section 5)." (`TASK-DOMAINS.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence that the core model is a dual-encoder with BERT-based text encoding.
Pass 2 (targeted source scan): performed - Confirmed explicit BERT architecture statement in Section 4.1.
