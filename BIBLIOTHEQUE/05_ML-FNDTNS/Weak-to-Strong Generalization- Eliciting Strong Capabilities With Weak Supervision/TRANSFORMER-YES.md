# Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision (2023)
Source: Weak-to-Strong Generalization- Eliciting Strong Capabilities With Weak Supervision.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The paper’s main experiments and training pipeline are built around GPT-4-family pretrained language models as both weak supervisors and strong students; this is the core model family for reported results.
- The paper explicitly states these models share GPT-4’s general architecture, which implies Transformer-style self-attention blocks are central to the method.

## Evidence
- "We test this using a range of pretrained language models in the GPT-4 family on natural language processing (NLP), chess, and reward modeling tasks." (Abstract, `Weak-to-Strong Generalization- Eliciting Strong Capabilities With Weak Supervision.md`)
- "These models share the same general architecture and pretraining dataset as GPT-4." (Section 1 footnote 1, `Weak-to-Strong Generalization- Eliciting Strong Capabilities With Weak Supervision.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for high-confidence YES; `Extending-dimensions analysis markdown` was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already established the central GPT-4-family architecture basis.
