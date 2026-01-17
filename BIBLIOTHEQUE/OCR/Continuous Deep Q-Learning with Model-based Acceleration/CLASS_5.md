# Continuous Deep Q-Learning with Model-based Acceleration (Not specified in the paper.)
Source: Continuous Deep Q-Learning with Model-based Acceleration.md

## Core reasons
- Proposes new reinforcement learning algorithms for continuous control (continuous Q-learning with NAF) and evaluates them, which is an ML methods contribution rather than data or benchmarks.
- Adds a model-based acceleration mechanism (imagination rollouts with fitted dynamics) to improve sample efficiency, focusing on training methodology and algorithmic design.

## Evidence extracts
- "We propose two complementary techniques for improving the efficiency of such algorithms. First, we derive a continuous variant of the Q-learning algorithm, which we call normalized adantage functions (NAF), as an alternative to the more commonly used policy gradient and actor-critic methods." (Abstract)
- "Adding these synthetic samples, which we refer to as *imagination rollouts*, to the replay buffer effectively augments the amount of experience available for Q-learning." (Section 5.2. Imagination Rollouts)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
