# Dota 2 with Large Scale Deep Reinforcement Learning (2021)
Source: Dota 2 with Large Scale Deep Reinforcement Learning.md

## Core reasons
- The paper centers on scaling reinforcement learning training and distributed systems to reach superhuman Dota 2 performance, emphasizing compute scale, batch size, and training time.
- It presents training methodology tools ("surgery") for continuing long-running RL training across environment/model changes rather than proposing new datasets or positional encoding methods.

## Evidence extracts
- "OpenAI Five leveraged existing reinforcement learning techniques, scaled to learn from batches of approximately 2 million frames every 2 seconds. We developed a distributed training system and tools for continual training which allowed us to train OpenAI Five for 10 months." (Abstract)
- "The key ingredients are to expand the scale of compute used, by increasing the batch size and total training time. In order to extend the training time of a single run to ten months, we developed surgery techniques for continuing training across changes to the model and environment." (6 Conclusion)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
