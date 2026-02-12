# Repeated examples help learn arithmetic (Year not specified)
Source: Repeated Examples Help Learn Arithmetic.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly states that the paper studies "small transformers" on the main tasks.
- The auxiliary analysis cites a sequence-to-sequence transformer with 8 attention heads as the model used for experiments.
- The extending-dimensions analysis markdown input was unavailable (`MISSING`), so the decision relies on the available abstract and auxiliary files.

## Evidence
- "We study small transformers trained on two problems of arithmetic: the greatest common divisor (GCD) and modular multiplication, and show that models trained on a limited set of repeated examples achieve better performance than models trained from unlimited data." (Abstract, `Repeated Examples Help Learn Arithmetic.md`)
- "We use sequence-to-sequence transformers (Vaswani et al., 2017) with 4 layers in the encoder and decoder, an embedding dimension of 512, and 8 attention heads (35 million trainable parameters)." (Section 2 Experimental settings, quoted in `TASK-DOMAINS.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for a high-confidence decision from the abstract and auxiliary files; extending-dimensions analysis markdown was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient.
