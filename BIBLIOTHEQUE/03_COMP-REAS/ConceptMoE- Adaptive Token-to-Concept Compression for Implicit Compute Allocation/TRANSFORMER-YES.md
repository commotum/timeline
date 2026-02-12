# ConceptMoE: Adaptive Token-to-Concept Compression for Implicit Compute Allocation (2026)
Source: ConceptMoE- Adaptive Token-to-Concept Compression for Implicit Compute Allocation.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes an LLM MoE method and explicitly reports reductions in "attention computation" and "KV cache," which are core Transformer self-attention mechanisms.
- Auxiliary files are consistent with attention-centric processing across tasks; the extending-dimensions analysis file was unavailable (`MISSING`) and was skipped.

## Evidence
- "Beyond performance, ConceptMoE reduces attention computation by up to  $R^2 \times$  and KV cache by  $R \times$ ." (Abstract, ConceptMoE- Adaptive Token-to-Concept Compression for Implicit Compute Allocation.md)
- "Based on the adaptive chunking and concept-representation architecture, attention is classified as dynamic and state as constructed across tasks" (Summary, TASK-DOMAINS.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence Transformer-based decision from abstract attention/KV-cache cues plus auxiliary attention characterization; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient.
