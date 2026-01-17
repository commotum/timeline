# RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning (Not specified in the paper.)
Source: RL$^2$- Fast Reinforcement Learning via Slow Reinforcement Learning.md

## Core reasons
- Proposes a fast RL algorithm implemented as an RNN whose weights are learned by a slow RL process, changing how computation is carried out for adaptation.
- Maintains recurrent state across episodes to adapt within a task, emphasizing a computational mechanism rather than data or positional encoding changes.

## Evidence extracts
- "In our proposed method, RL<sup>2</sup>, the algorithm is encoded in the weights of the RNN, which are learned slowly through a general-purpose ("slow") RL algorithm." (Abstract)
- "The RNN receives all information a typical RL algorithm would receive, including observations, actions, rewards, and termination flags; and it retains its state across episodes in a given Markov Decision Process (MDP)." (Abstract)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
