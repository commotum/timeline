# Efficient Evolutionary Program Synthesis (2025)
Source: Efficient Evolutionary Program Synthesis.md

## Core reasons
- Proposes an LLM-assisted evolutionary program synthesis system that grows and reuses a program library, changing how computation/search is performed to solve tasks.
- Focuses on algorithmic mechanisms for generating, selecting, and iteratively improving programs rather than datasets, positional encoding, or dimensional lifting.

## Evidence extracts
- "I decided to combine these two ideas: using LLMs to generate programs in Python (a Turing-complete language), growing system expertise by adding promising programs to a library, and including the current best program from the library in the LLM prompt to search for a better solution." (Section Motivation)
- "Starting from an empty library, my system loops through each task to prompt an LLM for Python program(s) that can solve all of the training examples." (Section Architecture)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
