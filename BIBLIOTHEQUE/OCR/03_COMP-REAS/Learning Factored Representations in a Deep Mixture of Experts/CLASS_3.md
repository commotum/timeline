# Learning Factored Representations in a Deep Mixture of Experts (Not specified in the paper.)
Source: Learning Factored Representations in a Deep Mixture of Experts.md

## Core reasons
- Proposes a Deep Mixture of Experts with layered gating to select expert combinations, changing computation by routing inputs through subsets of the model.
- Frames the goal as conditional computation to scale model capacity while keeping per-input computation low.

## Evidence extracts
- "we extend the Mixture of Experts to a stacked model, the Deep Mixture of Experts, with multiple sets of gating and experts." (Abstract)
- "use only a small portion of the network for each given input. Then, learn a computationally cheap mapping function from input to the appropriate portions of the network." (1 Introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
