# Sequence to Sequence Learning with Neural Networks (2014)
Source: Sequence to Sequence Learning with Neural Networks.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes an encoder-decoder built from deep LSTMs, not Transformer blocks or self-attention as the core architecture.
- Auxiliary task/domain files consistently tag attention as static and model state as constructed via LSTM encoder-decoder behavior, with no central Transformer-family cue.
- The Extending-dimensions analysis markdown was unavailable (`MISSING`), but the abstract plus available auxiliary files were sufficient for a high-confidence decision.

## Evidence
- "Our method uses a multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector." (Sequence to Sequence Learning with Neural Networks.md, Abstract, line 11)
- "Sequence-to-sequence machine translation (direct generation),Source-language sentence tokens,1D (t) (inferred),Open (inferred),Static (inferred),Constructed (inferred),Target-language sentence tokens,1D (t) (inferred),Open (inferred)" (TASK-DOMAINS.csv, line 2)
- "`Attention Dynamic` is `Static` because the described model encodes the full source into fixed vector `v` and decodes from it (Section 2), while attention is discussed as a mechanism used by other work (Section 1)." (TASK-DOMAINS.md, line 18)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for high-confidence TRANSFORMER-NO; Extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient.
