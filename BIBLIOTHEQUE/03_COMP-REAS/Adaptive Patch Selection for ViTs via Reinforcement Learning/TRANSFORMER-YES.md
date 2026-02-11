# Adaptive Patch Selection for ViTs via Reinforcement Learning (2025)
Source: TASK-DOMAINS.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: hint-only

## Why
- The hint files describe AgentViT as a method built around improving a Vision Transformer by selecting patches based on attention-derived state.
- The reported experiments and conclusions explicitly use ViT/SimpleViT as the central model family for main results.

## Evidence
- "an agent that selects the most important patches to improve the learning of a ViT." (TASK-DOMAINS.md, Abstract)
- "We tested AgentViT using ViT and SimpleViT as Vision Transformers, a Double Deep Q-Network as the internal agent, and applying it to CIFAR10, FashionMNIST, and Imagenette<sup>+</sup>." (TASK_MODEL_RATIO.md, Section 6 Conclusion)

## Pass accounting
Pass 0 (hint-first): performed - High-confidence TRANSFORMER-YES from explicit ViT-centric method and experiments in hint files.
Pass 1 (source triage): skipped - Hint evidence was sufficient for a confident binary decision.
Pass 2 (source deep dive): skipped - Not needed after Pass 0.
