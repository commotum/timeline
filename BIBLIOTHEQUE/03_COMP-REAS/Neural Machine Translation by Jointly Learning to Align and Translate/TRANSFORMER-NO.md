# Neural Machine Translation by Jointly Learning to Align and Translate (2014)
Source: Neural Machine Translation by Jointly Learning to Align and Translate.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes an encoder-decoder NMT model with RNN-based soft alignment, not Transformer self-attention blocks.
- Auxiliary analysis identifies RNN-centered architecture cues (bidirectional RNN encoder and RNN hidden states) and no Transformer-family model cues.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the available Pass 1 evidence was sufficient for a high-confidence decision.

## Evidence
- "The models proposed recently for neural machine translation often belong to a family of encoder—decoders and encode a source sentence into a fixed-length vector from which a decoder generates a translation." (Abstract, `Neural Machine Translation by Jointly Learning to Align and Translate.md`)
- "The new architecture consists of a bidirectional RNN as an encoder (Sec. 3.2) and a decoder that emulates searching through a source sentence during decoding a translation (Sec. 3.1)." (Section 3, `Neural Machine Translation by Jointly Learning to Align and Translate.md`)
- "s_i is an RNN hidden state for time i" (Section 3.1 Decoder: General Description, quoted in `TASK-DOMAINS.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence from abstract, `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; extending-dimensions input was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was already sufficient for a high-confidence binary decision.
