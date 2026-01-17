# Reward learning from human preferences and demonstrations in Atari (Not specified in the paper.)
Source: Reward learning from human preferences and demonstrations in Atari.md

## Core reasons
- Proposes a training methodology that combines human demonstrations and preference feedback to learn a reward model and train an RL agent without explicit rewards.
- Focuses on reward learning and policy training protocol in reinforcement learning rather than positional encoding, dimensional lifting, or dataset construction.

## Evidence extracts
- "In this work, we combine two approaches to learning from human feedback: expert demonstrations and trajectory preferences. We train a deep neural network to model the reward function and use its predicted reward to train an DQN-based deep reinforcement learning agent on 9 Atari games." (Abstract)
- "Our method for training the agent has the following components: an *expert* who provides demonstrations; an *annotator* (possibly the same as the expert) who gives preference feedback; a *reward model* that estimates a reward function from the annotator's preferences and, possibly, the demonstrations; and the *policy*, trained from the demonstrations and the reward provided by the reward model." (Section 2.2 The training protocol)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
