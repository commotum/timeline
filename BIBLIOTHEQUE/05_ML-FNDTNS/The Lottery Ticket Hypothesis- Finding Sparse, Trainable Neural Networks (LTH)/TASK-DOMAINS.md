# THE LOTTERY TICKET HYPOTHESIS: FINDING SPARSE, TRAINABLE NEURAL NETWORKS (Not specified in the paper.)
Source: The Lottery Ticket Hypothesis- Finding Sparse, Trainable Neural Networks (LTH).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | images (MNIST digits; CIFAR10 color images) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | labels (10 classes) (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers one task intent: image classification on MNIST and CIFAR10. The supported modality is image input to label output, with input indexed as 2D (x, y) and output as 0D under the glossary scheme. From the architecture and dataset descriptions, both input and output interfaces are fixed-size, and runtime behavior is consistent with static attention and direct state. These Dynamics/Attention/State values are inferred from feed-forward network descriptions rather than explicitly labeled in the paper.

## Evidence
### Task: Image classification
- "We only consider vision-centric classification tasks on smaller datasets (MNIST, CIFAR10)." (Section 6 LIMITATIONS AND FUTURE WORK)
- "The CIFAR10 dataset consists of 50,000 32x32 color (three-channel) training examples and 10,000 test examples." (Appendix H.1 EXPERIMENTAL METHODOLOGY)
- "This Section considers the fully-connected Lenet architecture (LeCun et al., 1998), which comprises two fully-connected hidden layers and a ten unit output layer, on the MNIST dataset." (Appendix G.1 EXPERIMENTAL METHODOLOGY)
- Inference: Input dimension is 2D (x, y) (inferred) because CIFAR10 is explicitly described as "32x32 color" images; input dynamics is Fixed (inferred) because the datasets use fixed image sizes. Attention Dynamic is Static (inferred) and State Dynamic is Direct (inferred) based on the paper's "dense, randomly-initialized, feed-forward networks" formulation. Output is labels with 0D and Fixed dynamics (inferred) based on the "ten unit output layer." (ABSTRACT; Appendix G.1; Appendix H.1)
