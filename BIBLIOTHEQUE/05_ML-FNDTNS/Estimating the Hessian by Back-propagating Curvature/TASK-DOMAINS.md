# Estimating the Hessian by Back-propagating Curvature (2012)
Source: Estimating the Hessian by Back-propagating Curvature.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hessian estimation (computational graph functions) | Real-valued vector input to scalar function f (computational graph) | 1D (t) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | Hessian matrix estimate of f | 2D (x, y) | Fixed (inferred) |
| Parameter estimation via score matching (MRF/cRBM) | Image patches (data) | 2D (x, y) | Fixed | Not specified in the paper. | Not specified in the paper. | Model parameters | 1D (t) | Fixed (inferred) |

## Summary
The paper primarily covers Hessian estimation for scalar functions defined by computational graphs, producing full-matrix (or entry-wise) Hessian estimates from vector inputs. The only explicit applied learning task is score-matching-based parameter estimation for a Markov random field/cRBM trained on fixed-size 16 x 16 image patches, yielding a fixed-size parameter vector. Attention and state dynamics are not specified in the paper.

## Evidence
### Task: Hessian estimation (computational graph functions)
- "develop Curvature Propagation (CP), a general technique for efficiently computing unbiased approximations of the Hessian" (Abstract)
- "of any function that is computed using a computational graph." (Abstract)
- "CP can give a rank-1 approximation of the whole Hessian" (Abstract)
- "Let  $f: \mathbb{R}^n \longrightarrow \mathbb{R}$  be a twice differentiable function." (Section 2.1)
- Inference: Marked input/output dynamics as Fixed (inferred) because the function is defined with fixed-size input $\mathbb{R}^n$ and its Hessian is $n \times n$. (Section 2.1)

### Task: Parameter estimation via score matching (MRF/cRBM)
- "we focus on estimating the parameters of a Markov random field using the score matching technique." (Section 8.2)
- "We trained the model on 11000 image patches of size  $16 \times 16$  from the Berkeley dataset  $^3$ ." (Section 8.2)
- "our cRBM contained 256 factors and hidden units." (Section 8.2)
- Inference: Marked output dynamics as Fixed (inferred) because the cRBM architecture is fixed ("256 factors and hidden units"), implying a fixed-size parameter vector. (Section 8.2)
