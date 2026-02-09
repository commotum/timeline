# Reward-Respecting Subtasks for Model-Based Reinforcement Learning (Not specified in the paper.)
Source: Reward-Respecting Subtasks for Model-Based Reinforcement Learning.md

## Core reasons
- The paper’s main contribution is a new algorithmic mechanism for temporal abstraction in model-based RL: reward-respecting subtasks within the STOMP pipeline, aimed at improving planning behavior.
- The work centers on how learned options/models change planning computation and efficiency, rather than introducing a dataset/benchmark or transformer positional/dimensional changes.

## Evidence extracts
- "The primary conceptual innovation of the current work is to introduce the notion of a reward-respecting subtask, that is, of a subtask that optimizes the rewards of the original task until terminating in a state that is sometimes of high value." (Section 1. The challenge of discovering temporal abstractions)
- "The primary definition of a useful option is one whose model takes the maximum in (18) or (19) at some state and thus makes a difference in planning." (Section 5. Planning with options)
- "Reward-respecting subtasks are a tiny subset of all possible subtasks, but they may be the most important in model-based reinforcement learning." (Section 8. Conclusions and future work)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
