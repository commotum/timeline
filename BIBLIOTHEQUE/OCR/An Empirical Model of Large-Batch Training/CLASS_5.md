# An Empirical Model of Large-Batch Training (Not specified in the paper.)
Source: An Empirical Model of Large-Batch Training.md

## Core reasons
- Proposes and empirically validates a training-optimization model using the gradient noise scale to predict the largest useful batch size and related efficiency tradeoffs.
- Focuses on general principles of large-batch training and optimization behavior across tasks rather than introducing new architectures, positional encodings, or datasets.

## Evidence extracts
- "we demonstrate that a simple and easy-to-measure statistic called the *gradient noise scale* predicts the largest useful batch size across many domains and applications, including a number of supervised learning datasets (MNIST, SVHN, CIFAR-10, ImageNet, Billion Word), reinforcement learning domains (Atari and Dota), and even generative model training (autoencoders on SVHN)." (Abstract)
- "The tradeoff between the speed and efficiency of neural network training is controlled by the batch size and follows the form of Equation 2.11." (Section 2.6 Summary)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
