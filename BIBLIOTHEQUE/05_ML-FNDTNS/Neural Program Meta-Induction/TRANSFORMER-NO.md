# Neural Program Meta-Induction (Year not specified)
Source: Neural Program Meta-Induction.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: source-targeted-scan

## Why
- The architecture used for the main results is CNN + LSTM (+ max-pooling for meta examples), with no Transformer-style self-attention blocks described.
- Auxiliary analyses also characterize attention as static/non-attention-centric, and the expected extending-dimensions file was unavailable (`MISSING`).

## Evidence
- "The input encoder is a 3-layer CNN with a FC+relu layer on top. The output decoder is a 1-layer LSTM. For the META model, the task encoder uses 1-layer CNN ... Multiple I/O examples were combined with max-pooling on the final vector." (Neural Program Meta-Induction.md, Section 8 Experimental Results, line 134)
- "Inference: Attention Dynamic labeled Static because the architecture processes the full grid with a CNN/LSTM and no runtime selection is described." (TASK-DOMAINS.md, Evidence section)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Strong non-Transformer signal from abstract framing plus TASK-DOMAINS/TASK-DOMAINS.csv/TASK_MODEL_RATIO; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): performed - Confirmed architecture cues in source and no Transformer/self-attention keyword evidence.
