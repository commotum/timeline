# ARC-AGI WITHOUT PRETRAINING (Year not specified)
Source: ARC-AGI Without Pretraining (CompressARC).md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: medium
Basis: hint-only

## Why
- The hint files describe the central trained model as an `equivariant_NN` with a residual backbone, not a Transformer/self-attention block stack.
- No hint file provides any explicit Transformer-style architecture cue (self-attention, ViT, BERT/GPT/LLaMA, Swin, Performer, RoFormer) for the main model.

## Evidence
- "Randomly initialize weights \theta for equivariant_NN_{\theta};" (TASK_MODEL_RATIO.md, Section 3.2 SEED OPTIMIZATION / Algorithm 3: CompressARC)
- "consists of a decoding layer functioning like an embedding matrix (details in Appendix D.1), followed by a core with a residual backbone" (TASK-DOMAINS.md, Section 4 ARCHITECTURE)

## Pass accounting
Pass 0 (hint-first): performed - hints were sufficient for a medium-confidence non-Transformer decision.
Pass 1 (source triage): skipped - avoided opening primary OCR source due to sufficient hint evidence.
Pass 2 (source deep dive): skipped - not needed after Pass 0.
