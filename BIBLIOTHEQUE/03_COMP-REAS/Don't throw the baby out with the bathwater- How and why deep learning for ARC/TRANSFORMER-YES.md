# Don't throw the baby out with the bathwater: How and why deep learning for ARC (Year not specified)
Source: Don't throw the baby out with the bathwater- How and why deep learning for ARC.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes a method built from pretrained LLMs with architecture/training choices centered on that model family, which strongly indicates a Transformer-based backbone.
- The auxiliary task-domain analysis explicitly cites non-causal encoder attention over the full sequence, i.e., Transformer-style self-attention used in the core ARC solver.
- Extending-dimensions analysis markdown was unavailable (`MISSING`), but the abstract plus available auxiliary files were sufficient for a high-confidence decision.

## Evidence
- "Concretely, we propose a methodology for training on ARC, starting from pretrained LLMs, and enhancing their ARC reasoning." (Don't throw the baby out with the bathwater- How and why deep learning for ARC.md, ABSTRACT)
- "non-causal (unmasked) attention within the encoder, allowing each token to simultaneously attend to the entire input sequence." (TASK-DOMAINS.md, Evidence: Grid-to-grid transformation, citing Section 3.2.2 Attention and masking)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for high-confidence TRANSFORMER-YES from abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md.
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient; extending-dimensions analysis file was unavailable (`MISSING`).
