# Multi-Game Decision Transformers (Not specified in the paper.)
Source: Multi-Game Decision Transformers.md

## Core reasons
- The paper's main contribution is a transformer-based offline RL approach for multi-game generalist agents and its scalability, which is a modeling/training contribution rather than a dataset or positional-encoding change.
- It explicitly formulates RL as sequence modeling and evaluates transformer-based decision transformers for performance across many games, fitting ML foundations/training methodology.

## Evidence extracts
- "We compare several approaches in this multi-game setting, such as online and offline RL methods and behavioral cloning, and find that our Multi-Game Decision Transformer models offer the best scalability and performance." (Abstract)
- "Following [14], we pose the problem of offline reinforcement learning as a sequence modeling problem where we model the probability of the next sequence token  $x_i$  conditioned on all tokens prior to it:  $P_{\theta}(x_i|x_{< i})$ , similar to contemporary decoder-only sequence models [12, 15, 62]." (Section 3.1 Reinforcement Learning as Sequence Modeling)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
