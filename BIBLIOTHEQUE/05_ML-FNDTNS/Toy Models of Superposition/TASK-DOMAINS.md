# Toy Models of Superposition (2022)
Source: Toy Models of Superposition.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reconstruction (autoencoding) of sparse feature vectors | Sparse synthetic feature vector x | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Reconstructed feature vector x' | 1D (t) (inferred) | Fixed (inferred) |
| Feature-wise absolute value computation | Sparse feature vector x with values in [-1, 1] | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Feature-wise absolute value vector y (target y=abs(x)) | 1D (t) (inferred) | Fixed (inferred) |

## Summary
The paper covers two distinct toy-model tasks: sparse feature-vector reconstruction (autoencoding) and feature-wise absolute value computation. In both cases, the model consumes and produces fixed-length vectors, which supports a 1D (t) and Fixed classification (inferred from the vector equations and setup). The architectures are feed-forward ReLU models with no runtime mechanism for selecting different input slices, so Attention is Static and State is Direct (both inferred). Other sections (phase changes, geometry, learning dynamics, adversarial robustness) analyze these tasks rather than introducing new task intents.

## Evidence
### Task: Reconstruction (autoencoding) of sparse feature vectors
- "Our goal is to explore whether a neural network can project a high dimensional vector  $x \in R^n$  into a lower dimensional vector  $h \in R^m$  and then recover it." (Section **Experiment Setup**)
- "Our first experiments will test the extent to which the idealized activations of an imagined larger model can be stored and recovered from a lower-dimensional space." (Section **Experiment Setup**)
- "This experiment setup could also be viewed as an autoencoder reconstructing x." (Footnote 5)
- Inference: `1D (t)`, `Fixed`, `Static`, and `Direct` are inferred from the fixed-length vector formulation and feed-forward mappings (`h = Wx`, `x' = W^Th + b` / `x' = ReLU(W^Th + b)`) with no runtime input-selection mechanism (Sections **THE MODEL (X o X')** and **Experiment Setup**).

### Task: Feature-wise absolute value computation
- "Specifically, we'll have the model compute  $y=\mathrm{abs}(x)$ ." (Section **Computation in Superposition**)
- "The target output y is y=abs(x)." (Section **Experiment Setup**)
- "Let's look at what happens when we train a model with n=3 features to perform absolute value on m=6 hidden layer neurons." (Section **Basic Results**)
- Inference: `1D (t)`, `Fixed`, `Static`, and `Direct` are inferred from vector input/output definitions and fixed feed-forward equations (`h = ReLU(W_1 x)`, `y' = ReLU(W_2 h + b)`) without dynamic runtime attention control or persistent constructed state (Section **Experiment Setup**).
