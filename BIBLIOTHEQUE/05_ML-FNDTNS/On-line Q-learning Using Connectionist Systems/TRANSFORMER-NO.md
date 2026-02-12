# On-line Q-learning Using Connectionist Systems (1994)
Source: On-line Q-learning Using Connectionist Systems.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes connectionist Q-learning with back-propagation neural networks and no self-attention/Transformer components.
- Auxiliary analyses describe per-action MLP networks (fixed inputs, hidden nodes, scalar outputs) and contain no Transformer-family architecture signals; the extending-dimensions file was unavailable.

## Evidence
- "In this report, the use of back-propagation neural networks (Rumelhart, Hinton and Williams 1986) is considered in this context." (On-line Q-learning Using Connectionist Systems.md, Abstract)
- "Neural networks, or *Multi-Layer Perceptrons*, provide such a continuous function approximation technique" (On-line Q-learning Using Connectionist Systems.md, Section 3 Connectionist Q-Learning)
- "The Q-function was represented by 6 neural networks, one for each available action." (TASK-DOMAINS.md, Evidence section: Task: prediction (action-value))
- "Each network had 26 inputs, 3 hidden nodes, and a single output," (TASK-DOMAINS.md, Evidence section: Task: prediction (action-value))

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for high-confidence TRANSFORMER-NO; Extending-dimensions analysis markdown was unavailable (MISSING).
Pass 2 (targeted source scan): skipped - Pass 1 was already decisive.
