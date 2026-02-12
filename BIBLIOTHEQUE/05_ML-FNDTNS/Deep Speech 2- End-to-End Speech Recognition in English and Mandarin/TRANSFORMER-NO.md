# Deep Speech 2: End-to-End Speech Recognition in English and Mandarin (Year not specified)
Source: Deep Speech 2- End-to-End Speech Recognition in English and Mandarin.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The auxiliary analysis identifies the central DS2 architecture as a recurrent neural network with convolution/recurrent components, not a Transformer or self-attention block stack.
- No material Transformer-style self-attention mechanism is indicated in the abstract or auxiliary files; the extending-dimensions file was unavailable (`MISSING`) and therefore skipped.

## Evidence
- "Figure 1 shows the architecture of the DS2 system which at its core is similar to the previous DS1 system [26]: a recurrent neural network (RNN) trained to ingest speech spectrograms and generate text transcriptions." (`TASK-DOMAINS.md` line 14, Evidence section)
- "The paper describes an end-to-end speech recognition system that maps speech spectrograms to text transcriptions for English and Mandarin. Inputs are time-frequency spectrograms (2D), and outputs are sequences of graphemes/characters (1D). The model operates over variable-length utterances and transcriptions (Open, inferred) and processes the full input sequence without any described dynamic attention or external state mechanism (Static attention and Direct state inferred)." (`TASK-DOMAINS.md` line 10, Summary section)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence NON-Transformer decision from abstract plus `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; extending-dimensions analysis was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided high-confidence architectural evidence.
