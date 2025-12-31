# Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective (2025)
Source: c8414b-2025.pdf

## Core reasons
- Proposes a search-and-scoring inference mechanism (DFS candidate generation plus probabilistic re-scoring) to solve ARC tasks, changing how computation is performed at inference time.
- Uses an LLM in dual roles (generator and scorer) with likelihood-based selection, emphasizing algorithmic reasoning procedures rather than new positional encodings or higher-dimensional inputs.

## Evidence extracts
- "we leverage task-specific data augmentations
throughout the training, generation, and scoring
phases, and employ a depth-first search algo-
rithm to generate diverse, high-probability can-
didate solutions." (p. 1)
- "Furthermore, we utilize the
LLMnotonlyasageneratorbutalsoasascorer,
using its output probabilities to select the most
promising solutions." (p. 1)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
