# MUPROP: UNBIASED BACKPROPAGATION FOR STOCHASTIC NEURAL NETWORKS (Not specified in the paper.)
Source: MuProp- Unbiased Backpropagation for Stochastic Neural Networks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| structured output prediction (MNIST lower-half completion) | top half of an MNIST digit | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Constructed (inferred) | lower half of an MNIST digit | 2D (x, y) (inferred) | Fixed (inferred) |
| structured output prediction (TFD facial expression prediction) | average face | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Constructed (inferred) | multiple facial expressions | 2D (x, y) (inferred) | Fixed (inferred) |
| generative modeling (variational training) | binarized MNIST images | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Constructed (inferred) | MNIST images (inferred) | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper evaluates MuProp on structured output prediction for image-like tasks (MNIST digit completion and TFD facial expression prediction) and on variational training of generative models for binarized MNIST. Across these tasks, the inputs and outputs are fixed-size, 2D spatial data inferred from pixelized datasets and fixed-width model configurations. The paper does not specify any attention mechanism, and it uses stochastic hidden or latent variables, implying constructed state.

## Evidence
### Task: structured output prediction (MNIST lower-half completion)
- "predict the lower half of an MNIST digit given the top half" (Section 5.1 Structured Output Prediction)
- "For MNIST, the output pixels are binarized" (Section 5.1 Structured Output Prediction)
- "Given an input x, an output y, and stochastic hidden variables h" (Section 5.1 Structured Output Prediction)
- Inference: Treated inputs/outputs as fixed 2D pixel grids because the task mentions "output pixels" and fixed-size MNIST models (e.g., "MNIST 392-200-200-392"); marked state as Constructed due to "stochastic hidden variables h". (Section 5.1 Structured Output Prediction)

### Task: structured output prediction (TFD facial expression prediction)
- "predict multiple facial expressions from an average face using Toronto Face dataset (TFD)" (Section 5.1 Structured Output Prediction)
- "Given an input x, an output y, and stochastic hidden variables h" (Section 5.1 Structured Output Prediction)
- "TFD 2034-200-200-2034" (Table 2, Section 5.1 Structured Output Prediction)
- Inference: Treated inputs/outputs as fixed 2D images because the task is about an "average face" and uses fixed-size TFD models ("TFD 2034-200-200-2034"); marked state as Constructed due to "stochastic hidden variables h". (Section 5.1 Structured Output Prediction)

### Task: generative modeling (variational training)
- "we apply MuProp to variational training of generative models" (Section 5.2 Variational training of generative models)
- "training layered belief networks with either Bernoulli or multinomial latent variables" (Section 5.2 Variational training of generative models)
- "binarized MNIST dataset, which consists of  $28\times28$  images of hand-written digits" (Section 5.2 Variational training of generative models)
- Inference: Labeled the domain as fixed 2D images based on the " $28\times28$  images" description and marked state as Constructed due to the stated "latent variables"; output images are inferred because the task is generative modeling over MNIST images. (Section 5.2 Variational training of generative models)
