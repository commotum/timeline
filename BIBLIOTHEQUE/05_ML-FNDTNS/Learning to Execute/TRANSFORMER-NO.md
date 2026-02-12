# LEARNING TO EXECUTE (Year not specified)
Source: Learning to Execute.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly states the central architecture is RNNs with LSTM units and describes sequence-to-sequence learning with LSTM.
- Auxiliary files (`TASK-DOMAINS.md`, `TASK_MODEL_RATIO.md`, and `TASK-DOMAINS.csv`) describe LSTM-based character-sequence tasks and provide no evidence of Transformer-style self-attention as a core model component.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the available abstract plus auxiliary evidence is already decisive.

## Evidence
- "Recurrent Neural Networks (RNNs) with Long Short-Term Memory units (LSTM) are widely used..." (Abstract, `Learning to Execute.md`)
- "Our main result is that LSTMs can learn to map the character-level representations of such programs to their correct outputs." (Abstract, `Learning to Execute.md`)
- "The paper evaluates sequence-to-sequence LSTMs on three character-level tasks..." (Summary, `TASK-DOMAINS.md`)
- "In both experiments, we used the same LSTM architecture." (Section 6 quote, `TASK_MODEL_RATIO.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for TRANSFORMER-NO from abstract and auxiliary analyses; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided high-confidence architecture identification.
