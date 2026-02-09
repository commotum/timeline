# Decoupling Search and Learning in Neural Net Training (Not specified in the paper.)
Source: Decoupling Search and Learning in Neural Net Training.md

## Core reasons
- The paper's central contribution is a training/optimization method that splits learning into search over representations and gradient-based regression to those targets.
- The empirical focus is on whether this training method approaches SGD on MNIST/CIFAR benchmarks, rather than proposing positional encoding changes, transformer dimensional adaptation, or a new benchmark resource.

## Evidence extracts
- "we propose a framework that performs training in two distinct phases" (Section ABSTRACT)
- "We design our training objective to ensure the convolutional body learns exclusively from the searched representations, not from classification gradients." (Section 4.2 Learning from Searched Representations)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
