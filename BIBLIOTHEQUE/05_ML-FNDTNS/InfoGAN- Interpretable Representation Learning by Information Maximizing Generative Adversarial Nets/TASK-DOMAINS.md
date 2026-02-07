# InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets (Not specified in the paper.)
Source: InfoGAN- Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image generation | latent code c and noise vector z | 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | images | 2D (x, y) (inferred) | Fixed (inferred) |
| Real/fake discrimination | images (real or generated) | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | real/fake probability D(x) | 0D (inferred) | Fixed (inferred) |
| Latent code inference | images | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | latent code distribution parameters Q(c|x) | 1D (t) (inferred) | Fixed (inferred) |

## Summary
The paper describes InfoGAN as a generative model that produces image samples from latent noise/codes and includes an adversarial discriminator plus an auxiliary network that predicts latent codes from images. The task coverage therefore spans image generation (latent vectors to images), real/fake discrimination (images to scalar scores), and latent code inference (images to latent code parameters). The dimensions span inferred 1D latent vectors and 2D image grids with fixed-size dynamics (inferred) for inputs and outputs, while attention and state dynamics are not specified in the paper.

## Evidence
### Task: Image generation
- "generator network G that generates samples from the generator distribution  $P_G$  by transforming a noise variable  $z \sim P_{\text{noise}}(z)$  into a sample G(z)." (Section 3 Background: Generative Adversarial Networks)
- "we provide the generator network with both the incompressible noise z and the latent code c, so the form of the generator becomes G(z,c)." (Section 4 Mutual Information for Inducing Latent Codes)
- "input noise vector into two parts: (i) z, which is treated as source of incompressible noise; (ii) c, which we will call the latent code" (Section 4 Mutual Information for Inducing Latent Codes)
- Inference: Treated the "noise vector" and latent code as fixed-length 1D inputs and the generated samples as 2D images; dimensions and dynamics are therefore marked inferred.

### Task: Real/fake discrimination
- "adversarial discriminator network D that aims to distinguish between samples from the true data distribution  $P_{\text{data}}(x)$  and the generator's distribution  $P_G$ ." (Section 3 Background: Generative Adversarial Networks)
- "the optimal discriminator is  $D(x) = P_{\text{data}}(x)/(P_{\text{data}}(x) + P_G(x))$ ." (Section 3 Background: Generative Adversarial Networks)
- Inference: Interpreted D(x) as a scalar real/fake score (0D) and the discriminator inputs as fixed-size images based on the discriminator formulation; dimensions and dynamics are therefore marked inferred.

### Task: Latent code inference
- "distribution Q(c|x) to approximate P(c|x):" (Section 5 Variational Mutual Information Maximization)
- "one final fully connected layer to output parameters for the conditional distribution Q(c|x)" (Section 6 Implementation)
- Inference: Interpreted the output of Q(c|x) as fixed-length 1D latent code parameters based on the "output parameters" description; dimensions and dynamics are therefore marked inferred.
