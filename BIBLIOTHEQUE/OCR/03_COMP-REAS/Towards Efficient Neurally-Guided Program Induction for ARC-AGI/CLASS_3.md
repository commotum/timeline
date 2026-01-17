# Towards Efficient Neurally-Guided Program Induction for ARC-AGI (Not specified in the paper.)
Source: Towards Efficient Neurally-Guided Program Induction for ARC-AGI.md

## Core reasons
- Proposes a neurally-guided program induction mechanism that enumerates and searches program space using transformer-produced token probabilities.
- Introduces an execution-guided transform-space approach to change how computation proceeds by conditioning on intermediate program states.

## Evidence extracts
- "We propose a novel probabilistic program enumerationbased search algorithm for program induction, leaning heavily on Transformer-based auto-regressive token sequences, rather than the typical n-gram approach, and analyze its strengths and weaknesses." (Section "Our main contributions")
- "The concept is to train a model such that, given an intermediate or starting program state, and a target grid, it predicts the probability distribution over the DSL for the next token." (Section "Learning the Transform Space")

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
