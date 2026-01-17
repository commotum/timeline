# An online sequence-to-sequence model for noisy speech recognition (Not specified in the paper.)
Source: An online sequence-to-sequence model for noisy speech recognition.md

## Core reasons
- The paper targets the limitation that standard sequence-to-sequence models require full input before producing outputs, and frames the need for online output.
- It centers on a mechanism that decides when to emit outputs via stochastic binary units with policy-gradient training improvements, changing how computation proceeds over time.

## Evidence extracts
- "Although remarkably successful, the sequence-to-sequence model with attention must process the entire input sequence before producing an output." (Section I. INTRODUCTION)
- "At each time step, i, a recurrent neural network (represented in figure 1) decides whether to emit an output token. The decision is made by a stochastic binary logistic unit  $b_i$ ." (Section II. METHODS)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
