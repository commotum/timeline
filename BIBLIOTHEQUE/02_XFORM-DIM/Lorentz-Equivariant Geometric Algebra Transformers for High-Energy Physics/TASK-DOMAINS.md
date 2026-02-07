# Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics (Not specified in the paper)
Source: Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Regression (QFT amplitude surrogate) | Phase-space particle four-momenta (initial and final particles) | 4D (x, y, z, t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Interaction amplitude (scalar) | 0D (inferred) | Fixed (inferred) |
| Classification (top tagging) | Reconstructed hadron point clouds (detector-level events) | 4D (x, y, z, t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Top-quark label (top vs background) | 0D (inferred) | Fixed (inferred) |
| Generation (reconstructed events) | Base distribution samples in Minkowski space / y coordinates | 4D (x, y, z, t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Reconstructed particle events (four-momenta) | 4D (x, y, z, t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates L-GATr on three particle-physics tasks: QFT amplitude regression, top-tagging classification, and generative modeling of reconstructed events. Inputs are particle-based representations (four-momenta) or base-distribution samples for flow-based generation, while outputs are scalar amplitudes/labels or generated particle events. The text supports 4D space-time representations and variable-size token sets, with explicit bounded multiplicities for the amplitude and generative datasets; attention is scaled dot-product and no constructed state beyond the input is described.

## Evidence
### Task: Regression (QFT amplitude surrogate)
- "We first demonstrate L-GATr as a neural surrogate for quantum field theoretical amplitudes" (Section 4.1 Surrogates for QFT amplitudes)
- "model to predict the amplitude as a function of the four-momenta of the initial and final particles." (Section 4.1 Surrogates for QFT amplitudes)
- "We generate training and evaluation data consisting of phase space inputs and their corresponding interaction amplitudes" (Appendix C.1 Dataset)
- "As example processes, we study  $q\bar{q}\to Z+ng$ , the production of a Z boson with  $n=1,\ldots,4$  additional gluons from a quark-antiquark pair." (Section 4.1 Surrogates for QFT amplitudes)
- "L-GATr represents high-energy data in a geometric algebra over four-dimensional space-time" (Abstract)
- "Because it computes pairwise interactions through scaled dot-product attention" (Section 1 Introduction)
- Inference: In Dimension marked 4D and Output marked 0D because the task uses particle "four-momenta" and predicts a single "amplitude." In Dynamics marked Capped because the processes specify $n=1,\ldots,4$ additional gluons. Attention Dynamic marked Static based on the scaled dot-product attention description. State Dynamic marked Direct because the task is described as predicting amplitudes directly from input four-momenta without any external state.

### Task: Classification (top tagging)
- "problem of classifying whether a spray of reconstructed hadrons originated from the decay of a top quark or any other process." (Section 4.2 Top tagging)
- "The data samples are structured as point clouds, with each event simulating a measurement by the ATLAS" (Appendix C.2 Dataset)
- "L-GATr is trained by minimizing a binary cross entropy (BCE) loss on the top quark labels." (Appendix C.2 Training)
- "We represent each particle as a token, store the particle type as a one-hot embedding in the scalar channels and the four-momentum in the first grade of the geometric algebra." (Section 3.1)
- "Because it computes pairwise interactions through scaled dot-product attention" (Section 1 Introduction)
- Inference: In Dimension marked 4D because particles are represented by four-momenta. Attention Dynamic marked Static based on scaled dot-product attention over the full token set. State Dynamic marked Direct because the task is framed as classification from inputs to labels without constructed external state. Out Dimension marked 0D and Out Dynamics marked Fixed because the output is a binary top-quark label.

### Task: Generation (reconstructed events)
- "Finally, we study the generative modelling of reconstructed events as an end-to-end generation task" (Section 4.3 Generative modelling)
- "We focus on the processes  $pp \to t\bar{t} + n$  jets, the generation of top pairs with  $n = 0 \dots 4$  additional jets" (Section 4.3 Generative modelling)
- "The base distribution is defined in the rescaled Minkowski space discussed above." (Appendix C.3 Base distribution)
- "The output of the L-GATr network is a vector field in Minkowski space  $(v_E, v_{p_x}, v_{p_y}, v_{p_z})$." (Appendix C.3 Models)
- "Because it computes pairwise interactions through scaled dot-product attention" (Section 1 Introduction)
- Inference: In/Out Dimension marked 4D because the generative model operates in Minkowski space $p=(E,p_x,p_y,p_z)$ and related four-dimensional coordinates. In/Out Dynamics marked Capped because the processes specify $n=0\dots4$ additional jets. Attention Dynamic marked Static based on scaled dot-product attention. State Dynamic marked Direct because the model is described as a direct generative mapping without persistent external state.
