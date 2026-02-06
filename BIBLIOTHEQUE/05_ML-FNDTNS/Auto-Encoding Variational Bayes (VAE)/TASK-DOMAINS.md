# Auto-Encoding Variational Bayes (Not specified in the paper)
Source: Auto-Encoding Variational Bayes (VAE).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Parameter estimation (ML/MAP learning) | Dataset of i.i.d. samples x | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Model parameters θ | Not specified in the paper. | Not specified in the paper. |
| Posterior inference / encoding (recognition/representation) | Datapoint x | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Distribution over latent code z | Not specified in the paper. | Not specified in the paper. |
| Generation / decoding (data generation) | Latent code z | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Datapoint x (generated data) | Not specified in the paper. | Not specified in the paper. |
| Marginal inference / density estimation | Datapoint x | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Marginal likelihood pθ(x) | Not specified in the paper. | Not specified in the paper. |
| Image denoising | Images | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Images | Not specified in the paper. | Not specified in the paper. |
| Image inpainting | Images | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Images | Not specified in the paper. | Not specified in the paper. |
| Image super-resolution | Images | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Images | Not specified in the paper. | Not specified in the paper. |
| Visualization / dimensionality reduction | High-dimensional data | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Low-dimensional latent manifold (e.g., 2D) | 2D (x, y) | Not specified in the paper. |

## Summary
The paper frames three core problems—parameter learning, posterior inference (encoding), and marginal inference—and describes the encoder/decoder as enabling data generation. It also cites applications in computer vision (image denoising, inpainting, super-resolution) and a visualization use case that projects high-dimensional data into a low-dimensional (e.g., 2D) latent space. Inputs and outputs are mostly described generically as datapoints x and latent codes z, with images explicitly named in experiments and applications. Dimension, dynamics, attention, and state are largely unspecified, with the only explicit dimensionality being the 2D latent space used for visualization.

## Evidence
### Task: Parameter estimation (ML/MAP learning)
- "Efficient approximate ML or MAP estimation for the parameters  $\theta$ ." (Section 2.1 Problem scenario)
- "we like to perform maximum likelihood (ML) or maximum a posteriori (MAP) inference on the (global) parameters" (Section 2 Method)

### Task: Posterior inference / encoding (recognition/representation)
- "Efficient approximate posterior inference of the latent variable z given an observed value x for a choice of parameters  $\theta$ ." (Section 2.1 Problem scenario)
- "since given a datapoint  $\mathbf{x}$  it produces a distribution (e.g. a Gaussian) over the possible values of the code  $\mathbf{z}$" (Section 2.1 Problem scenario)

### Task: Generation / decoding (data generation)
- "They also allow us to mimic the hidden random process and generate artificial data that resembles the real data." (Section 2.1 Problem scenario)
- "since given a code  $\mathbf{z}$  it produces a distribution over the possible corresponding values of  $\mathbf{x}$ ." (Section 2.1 Problem scenario)

### Task: Marginal inference / density estimation
- "Efficient approximate marginal inference of the variable x." (Section 2.1 Problem scenario)
- "The marginal likelihood is composed of a sum over the marginal likelihoods of individual datapoints" (Section 2.2 The variational bound)

### Task: Image denoising
- "Common applications in computer vision include image denoising, inpainting and super-resolution." (Section 2.1 Problem scenario)

### Task: Image inpainting
- "Common applications in computer vision include image denoising, inpainting and super-resolution." (Section 2.1 Problem scenario)

### Task: Image super-resolution
- "Common applications in computer vision include image denoising, inpainting and super-resolution." (Section 2.1 Problem scenario)

### Task: Visualization / dimensionality reduction
- "If we choose a low-dimensional latent space (e.g. 2D), we can use the learned encoders (recognition model)" (Section 5 Experiments)
- "to project high-dimensional data to a low-dimensional manifold." (Section 5 Experiments)
