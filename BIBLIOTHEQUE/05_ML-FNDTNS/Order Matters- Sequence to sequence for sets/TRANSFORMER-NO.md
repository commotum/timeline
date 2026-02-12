# ORDER MATTERS: SEQUENCE TO SEQUENCE FOR SETS (Year not specified)
Source: Order Matters- Sequence to sequence for sets.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract frames the work as seq2seq built on recurrent neural networks/LSTMs, not Transformer self-attention blocks.
- Auxiliary analysis describes the core systems as LSTM encoder/decoder and pointer-network variants with attention, not Transformer-style blocks.
- Extending-dimensions analysis markdown was unavailable (`MISSING`), but the available evidence is sufficient for a high-confidence decision.

## Evidence
- "Sequences have become first class citizens in supervised learning thanks to the resurgence of recurrent neural networks." (Order Matters- Sequence to sequence for sets.md, Abstract, line 11)
- "We trained medium sized LSTMs with large amounts of regularization" (TASK-DOMAINS.md, Evidence -> Task: Language modeling, line 24)
- "a sentence encoder LSTM followed by a decoder LSTM trained to generate a depth first traversal encoding of the parse tree, using an attention mechanism." (TASK-DOMAINS.md, Evidence -> Task: Constituency parsing, line 30)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence from abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md; Extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already gave a high-confidence non-Transformer determination.
