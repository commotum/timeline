# Adaptive patch selection to improve Vision Transformers through Reinforcement Learning (2025)
Source: bc06e1-2025.pdf

## Core reasons
- Introduces AgentViT, an RL-based framework that selects important patches so fewer patches are processed, explicitly targeting reduced computational load in ViTs while maintaining performance.
- Describes an adaptive patch-selection mechanism that changes which patches flow through the remaining ViT layers, i.e., variable computation driven by a learned agent.

## Evidence extracts
- "w e p ropose a ne w frame w o rk, called A gentV i T , which u ses R einforcement Learning to train a n agent that selects t he most important patches to impro v e the l earning of a V i T . The goal of AgentV i T i s t o r educe the number of patches processed b y a V i T , and thus its computational load, while still maintaining competiti v e performance." (p. 1)
- "(ii) The a gent computes Q-v a lues for each patch a nd only p atches with Q-v a lues greater than the m ean are s elected. (iii) The s elected patches a re propag a ted t hrough the r emaining V i T l ayers." (p. 7)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
