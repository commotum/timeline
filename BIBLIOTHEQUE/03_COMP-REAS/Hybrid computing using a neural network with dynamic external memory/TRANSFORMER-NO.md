# Hybrid computing using a neural network with dynamic external memory (2016)
Source: Hybrid computing using a neural network with dynamic external memory.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes a Differentiable Neural Computer with external read/write memory and does not present Transformer-style self-attention blocks as the central architecture.
- Auxiliary files characterize the model family as DNC/LSTM-style memory-augmented neural networks; the extending-dimensions file was unavailable (`MISSING`), but available evidence is still sufficient and consistent.

## Evidence
- "Here we introduce a machine learning model called a differentiable neural computer (DNC), which consists of a neural network that can read from and write to an external memory matrix" (Abstract, `Hybrid computing using a neural network with dynamic external memory.md`)
- "Any neural network can be used for the controller, but we have used the following variant of the deep LSTM architecture" (Methods excerpt captured in source markdown, `Hybrid computing using a neural network with dynamic external memory.md`)
- "Inference: Attention Dynamic and State Dynamic are marked Dynamic/Constructed because a DNC 'uses differentiable attention mechanisms' and 'the memory can be selectively written to as well as read.'" (Evidence section, `TASK-DOMAINS.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - high-confidence non-Transformer decision from abstract + TASK-DOMAINS/TASK-DOMAINS.csv/TASK_MODEL_RATIO, with extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient for a high-confidence decision.
