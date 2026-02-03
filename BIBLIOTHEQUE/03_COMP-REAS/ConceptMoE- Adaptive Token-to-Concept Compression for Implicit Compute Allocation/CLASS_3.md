# ConceptMoE: Adaptive Token-to-Concept Compression for Implicit Compute Allocation (2026)
Source: ConceptMoE- Adaptive Token-to-Concept Compression for Implicit Compute Allocation.md

## Core reasons
- Proposes an adaptive computation mechanism that merges tokens into concept representations to allocate compute implicitly, changing how inference is performed.
- Introduces learnable chunking to compress sequences before the compute-intensive concept model, focusing on computation allocation rather than positional encoding or datasets.

## Evidence extracts
- "We introduce ConceptMoE, which dynamically merges semantically similar tokens into concept representations, performing implicit token-level compute allocation." (Abstract)
- "A learnable chunk module identifies optimal boundaries by measuring inter-token similarity, compressing sequences before they enter the compute-intensive concept model." (Introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
