# An online sequence-to-sequence model for noisy speech recognition (Year not specified)
Source: An online sequence-to-sequence model for noisy speech recognition.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: hint-only

## Why
- The hint files describe the main experimental model as a 2-layer LSTM, with no Transformer-style self-attention architecture in the core method.
- The task/domain hints characterize the method as recurrent sequence processing rather than self-attention blocks.

## Evidence
- "The models we trained on TIMIT had two layers with 256 units per layer." (TASK_MODEL_RATIO.md, Section III-A quote)
- "Our model was a 2-layer LSTM with 256 units in each layer." (TASK_MODEL_RATIO.md, Section III-B quote)

## Pass accounting
Pass 0 (hint-first): performed - sufficient evidence for high-confidence NON-transformer classification from TASK_MODEL_RATIO.md and TASK-DOMAINS.md.
Pass 1 (source triage): skipped - hint evidence already decisive.
Pass 2 (source deep dive): skipped - not needed after decisive hint-only evidence.
