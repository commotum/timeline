# A Neurosymbolic Approach to Abstraction and Reasoning (2021)
Source: A Neurosymbolic Approach to Abstraction and Reasoning.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: source-triage

## Why
- The paper's central model is a bidirectional execution-guided program-synthesis agent with deep-set/pointer-network components, not a Transformer architecture.
- The model for reported results uses convolutional embeddings; Transformer-family terms appear only as background references, not as the core method.

## Evidence
- "Our network architecture mirrors that used by [6], with a network for embedding value nodes, a deep set embedding for the set of value nodes, and a pointer network for choosing arguments to an operation." (Section 4.3 Network and training)
- "We use a convolutional neural network to embed grid example sets." (Section 4.4 Results)
- "The network outputs a distribution over the set of functions using a convolutional network over the input/output grids." (Section 3 approach description)

## Pass accounting
Pass 0 (hint-first): performed - Hints established an ARC/24-game program-synthesis setup but did not explicitly confirm whether Transformer self-attention was part of the central model.
Pass 1 (source triage): performed - Architecture-level source scan found pointer-network/deep-set/CNN model descriptions and no Transformer/self-attention blocks used for main results.
Pass 2 (source deep dive): skipped - Pass 1 provided sufficient high-confidence evidence for a non-Transformer central model.
