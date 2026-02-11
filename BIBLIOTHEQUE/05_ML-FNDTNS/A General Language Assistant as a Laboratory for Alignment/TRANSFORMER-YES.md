# A General Language Assistant as a Laboratory for Alignment (Year not specified)
Source: A General Language Assistant as a Laboratory for Alignment.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-triage

## Why
- The paper explicitly states that the studied core models are decoder-only Transformer language models used throughout the experiments.
- Appendix architecture details confirm all trained base models are Transformer models, indicating self-attention is central rather than peripheral.

## Evidence
- "Throughout this paper we will be studying a consistent set of decoder-only Transformer language models with parameter counts ranging from about 10M to 52B" (Models section, line 179, A General Language Assistant as a Laboratory for Alignment.md)
- "All the decoder-only [LSP+18] Transformer [VSP+17] models we train have a fixed aspect ratio" (Appendix A Language Model Pre-training, line 585, A General Language Assistant as a Laboratory for Alignment.md)

## Pass accounting
Pass 0 (hint-first): performed - hints showed language-model setup but did not explicitly confirm Transformer/self-attention architecture.
Pass 1 (source triage): performed - found explicit statements that the main models are decoder-only Transformer language models.
Pass 2 (source deep dive): skipped - Pass 1 provided high-confidence direct architectural evidence.
