# On the difficulty of training recurrent neural networks (2013)
Source: On the difficulty of training recurrent neural networks (Pascanu, Mikolov & Bengio).md

## Core reasons
- The paper focuses on core training/optimization challenges for recurrent neural networks, analyzing vanishing and exploding gradients as a foundational learning issue.
- It proposes general training remedies (gradient norm clipping and a regularization term) rather than new architectures, datasets, or positional mechanisms.

## Evidence extracts
- "There are two widely known issues with properly training recurrent neural networks, the vanishing and the exploding gradient problems detailed in Bengio et al. (1994). In this paper we attempt to improve the understanding of the underlying issues by exploring these problems from an analytical, a geometric and a dynamical systems perspective. Our analysis is used to justify a simple yet effective solution. We propose a gradient norm clipping strategy to deal with exploding gradients and a soft constraint for the vanishing gradients problem." (Abstract)
- "We put forward a hypothesis stating that when gradients explode we have a cliff-like structure in the error surface and devise a simple solution based on this hypothesis, clipping the norm of the exploded gradients. ... In order to deal with the vanishing gradient problem we use a regularization term that forces the error signal not to vanish as it travels back in time." (5. Summary and conclusions)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
