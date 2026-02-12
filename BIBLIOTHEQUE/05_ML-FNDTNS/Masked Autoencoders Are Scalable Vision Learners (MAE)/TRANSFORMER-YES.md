# Masked Autoencoders Are Scalable Vision Learners (MAE) (Year not specified)
Source: Masked Autoencoders Are Scalable Vision Learners (MAE).md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly identifies the core model as a "vanilla ViT-Huge model," and ViT is a Transformer family architecture with self-attention blocks.
- The auxiliary analyses are consistent with token-based encoder/decoder modeling around masked patches; the extending-dimensions file was unavailable (`MISSING`) but was not needed to finalize.

## Evidence
- "Our scalable approach allows for learning high-capacity models that generalize well: e.g., a vanilla ViT-Huge model achieves the best accuracy (87.8%) among methods that use only ImageNet-1K data." (Abstract, `Masked Autoencoders Are Scalable Vision Learners (MAE).md`)
- "the full set of encoded patches and mask tokens is processed by a small decoder" (Quoted in `TASK-DOMAINS.md`, Evidence section, from Figure 1 caption)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence TRANSFORMER-YES decision from abstract ViT cue plus auxiliary consistency; extending-dimensions input was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was already sufficient.
