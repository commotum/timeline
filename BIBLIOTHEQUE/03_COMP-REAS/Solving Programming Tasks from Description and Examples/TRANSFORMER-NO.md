# NEURAL PROGRAM SEARCH: SOLVING PROGRAMMING TASKS FROM DESCRIPTION AND EXAMPLES (Year not specified)
Source: Solving Programming Tasks from Description and Examples.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The central model is Seq2Tree with an RNN encoder and a doubly-recurrent tree decoder; this is not Transformer-style self-attention architecture.
- Attention is present as decoder augmentation over encoded inputs, but the evidence does not indicate Transformer blocks or self-attention as the core model mechanism; the extending-dimensions file was unavailable (`MISSING`).

## Evidence
- "The encoder uses RNN to embed concatenation of arguments Args and tokenized textual description of the task Text." (Section 3.2 SeQ2Tree, quoted in `TASK-DOMAINS.md`)
- "The decoder is a doubly-recurrent neural network for generating tree structured output" (Section 3.2 SeQ2Tree, quoted in `TASK-DOMAINS.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - High-confidence NON-Transformer decision from abstract + TASK-DOMAINS.md + TASK-DOMAINS.csv + TASK_MODEL_RATIO.md; extending-dimensions analysis file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided sufficient architecture evidence.
