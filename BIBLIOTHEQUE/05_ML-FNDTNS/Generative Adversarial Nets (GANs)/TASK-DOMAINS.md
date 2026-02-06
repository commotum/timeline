# Generative Adversarial Nets (Not specified in the paper.)
Source: Generative Adversarial Nets (GANs).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generation | noise variables z | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | data samples x (e.g., natural images) | 2D (x, y) (inferred) | Fixed (inferred) |
| classification (real vs generated) | data samples x | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | probability that x came from data (scalar) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper defines a generative adversarial framework with two tasks: generating data samples from noise and discriminating real versus generated samples. The data domain explicitly includes natural images, so outputs are image-like samples produced by the generator. Based on the multilayer perceptron description, the inputs and outputs are fixed-size with static attention and direct state (inferred).

## Evidence
### Task: generation
- "a generative model G that captures the data distribution" (Abstract)
- "define a prior on input noise variables  $p_{\boldsymbol{z}}(\boldsymbol{z})$ , then represent a mapping to data space as  $G(\boldsymbol{z};\theta_g)$" (Section 3 Adversarial nets)
- "The generator G implicitly defines a probability distribution  $p_g$  as the distribution of the samples G(z) obtained when  $z \sim p_z$ ." (Section 4 Theoretical Results)
- "natural images, audio waveforms containing speech, and symbols in natural language corpora." (Section 1 Introduction)
- Inference: Inferred 1D noise input, fixed dynamics, static attention, and direct state from the description of G as a multilayer perceptron mapping z to data space; inferred 2D outputs because the data domain includes natural images. (Sections 1 and 3)

### Task: classification (real vs generated)
- "define a second multilayer perceptron  $D(\boldsymbol{x};\theta_d)$  that outputs a single scalar." (Section 3 Adversarial nets)
- "$D(\boldsymbol{x})$  represents the probability that  $\boldsymbol{x}$  came from the data rather than  $p_g$ ." (Section 3 Adversarial nets)
- "We train D to maximize the probability of assigning the correct label to both training examples and samples from G." (Section 3 Adversarial nets)
- Inference: Inferred 2D inputs, fixed dynamics, static attention, and direct state because D is a multilayer perceptron over x; inferred 0D output from the "single scalar" output. (Section 3)
