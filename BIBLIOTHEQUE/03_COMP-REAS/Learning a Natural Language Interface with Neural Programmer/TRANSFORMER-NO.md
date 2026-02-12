# Learning a Natural Language Interface with Neural Programmer (Year not specified)
Source: Learning a Natural Language Interface with Neural Programmer.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: source-targeted-scan

## Why
- The central model is Neural Programmer with RNN components (question LSTM + history RNN) and discrete operations, not Transformer self-attention blocks.
- The paper’s attention mechanism is soft attention over question tokens, which is not presented as Transformer-style multi-head self-attention architecture.
- Extending-dimensions analysis markdown was unavailable (`MISSING`), but the abstract plus available auxiliary files and targeted architecture scan were sufficient for a high-confidence decision.

## Evidence
- "We enhance the objective function of Neural Programmer, a neural network with built-in discrete operations." (Learning a Natural Language Interface with Neural Programmer.md, ABSTRACT)
- "We use an LSTM network (Hochreiter & Schmidhuber, 1997) as the question RNN." (Learning a Natural Language Interface with Neural Programmer.md, Section 2 NEURAL PROGRAMMER)
- "attention vector obtained by performing soft attention (Bahdanau et al., 2014) on the question using the history vector." (TASK-DOMAINS.md, Evidence)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Read abstract, `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; no Transformer-model indicators were found.
Pass 2 (targeted source scan): performed - Scanned model description to confirm RNN/LSTM + soft-attention architecture and finalize the binary label.
