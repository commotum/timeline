# Adaptive Computation Time for Recurrent Neural Networks (Year not specified)
Source: TASK_MODEL_RATIO.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: hint-only

## Why
- Hint evidence identifies the core experimental architectures as simple RNNs and LSTMs, with no Transformer-style self-attention blocks.
- The method focus is adaptive computation for recurrent networks, so Transformer mentions are not central to the model used for main results.

## Evidence
- "The network architecture was a simple RNN with a single hidden layer containing  $128 \ tanh$  units and a single sigmoidal output unit, trained with binary cross-entropy loss on minibatches of size 128." (TASK_MODEL_RATIO.md, Section 3.1 Parity)
- "LSTM networks were used with a single layer of 1500 cells and a size 256 softmax classification layer." (TASK_MODEL_RATIO.md, Section 3.5 Wikipedia Character Prediction)

## Pass accounting
Pass 0 (hint-first): performed - Hints provided enough evidence that the central models are recurrent (RNN/LSTM), not Transformer/self-attention.
Pass 1 (source triage): skipped - High-confidence decision from hint files.
Pass 2 (source deep dive): skipped - Not needed after Pass 0.
