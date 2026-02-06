# An Empirical Model of Large-Batch Training (Not specified in the paper.)
Source: An Empirical Model of Large-Batch Training.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification (MNIST) | images (MNIST handwritten digits) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | class labels (inferred) | 0D (inferred) | Not specified in the paper. |
| Image classification (SVHN) | images (Street View House Numbers) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | class labels (inferred) | 0D (inferred) | Not specified in the paper. |
| Image classification (CIFAR10) | images (CIFAR10) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | class labels (inferred) | 0D (inferred) | Not specified in the paper. |
| Image classification (ImageNet) | images (ImageNet) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | class labels (inferred) | 0D (inferred) | Not specified in the paper. |
| Image autoencoding (SVHN autoencoder) | images (SVHN) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | images (reconstructions) (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |
| Variational autoencoding (SVHN VAE) | images (SVHN) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | images (reconstructions) (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |
| Language modeling (Billion Word, autoregressive prediction) | tokens (20-token sequences) | 1D (t) (inferred) | Fixed (20-token sequences) | Not specified in the paper. | Constructed (inferred) | token (last-token prediction) (inferred) | 0D (inferred) | Fixed (inferred) |
| Reinforcement learning control (Atari: Alien) | observations (Atari game) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | actions (inferred) | Not specified in the paper. | Not specified in the paper. |
| Reinforcement learning control (Atari: Beamrider) | observations (Atari game) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | actions (inferred) | Not specified in the paper. | Not specified in the paper. |
| Reinforcement learning control (Atari: Breakout) | observations (Atari game) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | actions (inferred) | Not specified in the paper. | Not specified in the paper. |
| Reinforcement learning control (Atari: Pong) | observations (Atari game) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | actions (inferred) | Not specified in the paper. | Not specified in the paper. |
| Reinforcement learning control (Atari: Qbert) | observations (Atari game) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | actions (inferred) | Not specified in the paper. | Not specified in the paper. |
| Reinforcement learning control (Atari: Seaquest) | observations (Atari game) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | actions (inferred) | Not specified in the paper. | Not specified in the paper. |
| Reinforcement learning control (Atari: Space Invaders) | observations (Atari game) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | actions (inferred) | Not specified in the paper. | Not specified in the paper. |
| Reinforcement learning control (Dota 1v1) | observations (Dota game) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | actions (inferred) | Not specified in the paper. | Not specified in the paper. |
| Reinforcement learning control (Dota 5v5) | observations (Dota game) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | actions (inferred) | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper covers image classification on MNIST, SVHN, CIFAR10, and ImageNet; generative image modeling on SVHN (autoencoder and VAE); language modeling on the Billion Word corpus; and reinforcement learning control in Atari and Dota environments. The only explicit structural constraint is the language model's fixed 20-token sequences and the use of LSTM state; image inputs/labels and autoencoder reconstructions are inferred from task descriptions. For RL, the paper specifies games and observations but does not detail observation/action dimensionality or attention/state dynamics, so most of those fields remain unspecified.

## Evidence
### Task: Image classification (MNIST)
- "For image classification, we use the following datasets:" (Section A.4.1 Classification)
- "- MNIST handwritten digits [LC10]" (Section A.4.1 Classification)
- Inference: Because the task is framed as "image classification," I inferred image inputs with 2D (x, y) structure and 0D class-label outputs. (Section A.4.1 Classification)

### Task: Image classification (SVHN)
- "For image classification, we use the following datasets:" (Section A.4.1 Classification)
- "- Street View House Numbers (SVHN) [NWC+11]" (Section A.4.1 Classification)
- Inference: Because the task is framed as "image classification," I inferred image inputs with 2D (x, y) structure and 0D class-label outputs. (Section A.4.1 Classification)

### Task: Image classification (CIFAR10)
- "For image classification, we use the following datasets:" (Section A.4.1 Classification)
- "- CIFAR10 [Kri09]" (Section A.4.1 Classification)
- Inference: Because the task is framed as "image classification," I inferred image inputs with 2D (x, y) structure and 0D class-label outputs. (Section A.4.1 Classification)

### Task: Image classification (ImageNet)
- "For image classification, we use the following datasets:" (Section A.4.1 Classification)
- "- ImageNet [DDS+09]" (Section A.4.1 Classification)
- Inference: Because the task is framed as "image classification," I inferred image inputs with 2D (x, y) structure and 0D class-label outputs. (Section A.4.1 Classification)

### Task: Image autoencoding (SVHN autoencoder)
- "VAE and Autoencoder We train a VAE [KW13] and a simple Autoencoder on the SVHN dataset [NWC+11]" (Section Generative Modeling)
- "we also provide training data on a simple autoencoder with the same architecture." (Section A.4.3 Generative and Language Modeling)
- Inference: The use of an autoencoder on SVHN implies image inputs and image reconstructions, so I labeled both input and output as 2D (x, y) images. (Sections Generative Modeling; A.4.3 Generative and Language Modeling)

### Task: Variational autoencoding (SVHN VAE)
- "VAE and Autoencoder We train a VAE [KW13] and a simple Autoencoder on the SVHN dataset [NWC+11]" (Section Generative Modeling)
- "we train a Variational Autoencoder [KW13] using the InfoGAN architecture [CDH+16] (see their appendix C.2) on the SVHN dataset." (Section A.4.3 Generative and Language Modeling)
- Inference: The VAE is used for generative image modeling on SVHN, which implies image inputs and image reconstructions; I therefore marked input/output as 2D (x, y) images. (Sections Generative Modeling; A.4.3 Generative and Language Modeling)

### Task: Language modeling (Billion Word, autoregressive prediction)
- "Language Modeling We train a single-layer LSTM for autoregressive prediction on the Billion Word dataset [CMS+13]" (Section Generative Modeling)
- "with 20-token sequences." (Section A.4.3 Generative and Language Modeling)
- "LSTM cell states were reset to zero between samples" (Section A.4.3 Generative and Language Modeling)
- Inference: I mapped token sequences to 1D (t), inferred constructed state from the LSTM cell state, and inferred single-token outputs from the autoregressive prediction setup. (Sections Generative Modeling; A.4.3 Generative and Language Modeling)

### Task: Reinforcement learning control (Atari: Alien)
- "seven Atari games [BNVB12] (Alien, Beamrider, Breakout, Pong, Qbert, Seaquest, Space Invaders)" (Section Reinforcement Learning)
- "Batch sizes reported in number of images, tokens (for language models), or observations (for games)." (Figure 4 caption)
- Inference: Because these are "RL agents" trained with a policy gradient algorithm, I inferred action outputs for the control task. (Section Reinforcement Learning)

### Task: Reinforcement learning control (Atari: Beamrider)
- "seven Atari games [BNVB12] (Alien, Beamrider, Breakout, Pong, Qbert, Seaquest, Space Invaders)" (Section Reinforcement Learning)
- "Batch sizes reported in number of images, tokens (for language models), or observations (for games)." (Figure 4 caption)
- Inference: Because these are "RL agents" trained with a policy gradient algorithm, I inferred action outputs for the control task. (Section Reinforcement Learning)

### Task: Reinforcement learning control (Atari: Breakout)
- "seven Atari games [BNVB12] (Alien, Beamrider, Breakout, Pong, Qbert, Seaquest, Space Invaders)" (Section Reinforcement Learning)
- "Batch sizes reported in number of images, tokens (for language models), or observations (for games)." (Figure 4 caption)
- Inference: Because these are "RL agents" trained with a policy gradient algorithm, I inferred action outputs for the control task. (Section Reinforcement Learning)

### Task: Reinforcement learning control (Atari: Pong)
- "seven Atari games [BNVB12] (Alien, Beamrider, Breakout, Pong, Qbert, Seaquest, Space Invaders)" (Section Reinforcement Learning)
- "Batch sizes reported in number of images, tokens (for language models), or observations (for games)." (Figure 4 caption)
- Inference: Because these are "RL agents" trained with a policy gradient algorithm, I inferred action outputs for the control task. (Section Reinforcement Learning)

### Task: Reinforcement learning control (Atari: Qbert)
- "seven Atari games [BNVB12] (Alien, Beamrider, Breakout, Pong, Qbert, Seaquest, Space Invaders)" (Section Reinforcement Learning)
- "Batch sizes reported in number of images, tokens (for language models), or observations (for games)." (Figure 4 caption)
- Inference: Because these are "RL agents" trained with a policy gradient algorithm, I inferred action outputs for the control task. (Section Reinforcement Learning)

### Task: Reinforcement learning control (Atari: Seaquest)
- "seven Atari games [BNVB12] (Alien, Beamrider, Breakout, Pong, Qbert, Seaquest, Space Invaders)" (Section Reinforcement Learning)
- "Batch sizes reported in number of images, tokens (for language models), or observations (for games)." (Figure 4 caption)
- Inference: Because these are "RL agents" trained with a policy gradient algorithm, I inferred action outputs for the control task. (Section Reinforcement Learning)

### Task: Reinforcement learning control (Atari: Space Invaders)
- "seven Atari games [BNVB12] (Alien, Beamrider, Breakout, Pong, Qbert, Seaquest, Space Invaders)" (Section Reinforcement Learning)
- "Batch sizes reported in number of images, tokens (for language models), or observations (for games)." (Figure 4 caption)
- Inference: Because these are "RL agents" trained with a policy gradient algorithm, I inferred action outputs for the control task. (Section Reinforcement Learning)

### Task: Reinforcement learning control (Dota 1v1)
- "- Dota 1v1 and 5v5 [BCD+18]" (Section A.4.2 Reinforcement Learning)
- "train PPO [SWD+17] agents on both Dota 1v1 and 5v5 environments" (Section Reinforcement Learning)
- "Batch sizes reported in number of images, tokens (for language models), or observations (for games)." (Figure 4 caption)
- Inference: Because the paper discusses training PPO agents, I inferred action outputs for the control task. (Section Reinforcement Learning)

### Task: Reinforcement learning control (Dota 5v5)
- "- Dota 1v1 and 5v5 [BCD+18]" (Section A.4.2 Reinforcement Learning)
- "train PPO [SWD+17] agents on both Dota 1v1 and 5v5 environments" (Section Reinforcement Learning)
- "Batch sizes reported in number of images, tokens (for language models), or observations (for games)." (Figure 4 caption)
- Inference: Because the paper discusses training PPO agents, I inferred action outputs for the control task. (Section Reinforcement Learning)
