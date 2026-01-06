# Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (Not specified in the paper.)
Source: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero).md

## Core reasons
- Frames AlphaZero's decision making as a different search computation than minimax, focusing on subtree-averaging instead of alpha-beta evaluation.
- Highlights MCTS as the mechanism that manages neural-network approximation errors during search, indicating a reasoning/search process as the core contribution.

## Evidence extracts
- "AlphaZero uses a markedly different approach that averages over the position evaluations within a subtree, rather than computing the minimax evaluation of that subtree." (Section MCTS and Alpha-Beta Search)
- "MCTS averages over these approximation errors, which therefore tend to cancel out when evaluating a large subtree." (Section MCTS and Alpha-Beta Search)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
