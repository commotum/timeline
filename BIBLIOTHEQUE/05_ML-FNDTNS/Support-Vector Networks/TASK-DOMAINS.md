# Support-Vector Networks (1995)
Source: Support-Vector Networks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Binary classification (two-group pattern recognition) | Input vectors; labeled patterns | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Binary class label ({-1, 1}) | 0D (inferred) | Fixed (inferred) |
| Multiclass handwritten digit classification (one-vs-all) | Bit-mapped digit images | 2D (x, y) | Fixed | Static (inferred) | Constructed (inferred) | Digit class label (maximum output over ten classifiers) | 0D (inferred) | Fixed |

## Summary
The paper covers classification tasks: two-group pattern classification and a one-vs-all setup for handwritten digit recognition. Inputs are fixed-size vectors/patterns, including 2D bit-mapped digit images, and outputs are discrete class decisions. The supported dimensions are 1D (feature-indexed vectors, inferred) and 2D (pixel grids), with 0D outputs. The classifier uses a fixed learned support-vector expansion at inference (static attention, inferred) and depends on learned hyperplane/support-vector representations (constructed state, inferred).

## Evidence
### Task: Binary classification (two-group pattern recognition)
- "The support-vector network is a new learning machine for two-group classification problems." (Abstract)
- "The set of labeled training patterns

$$(y_1, \mathbf{x}_1), \dots, (y_\ell, \mathbf{x}_\ell), \quad y_i \in \{-1, 1\}$$" (Section 2.1)
- "Classification of an unknown vector x is done by first transforming the vector to the separating space  $(x \mapsto \phi(x))$  and then taking the sign of the function" (Section 4)
- Inference: `1D (t)` and `Fixed` are inferred from the explicit "n-dimensional input vector" formulation (`\phi : \mathfrak{R}^n \to \mathfrak{R}^N`, Section 4). `Static` attention is inferred from the fixed support-vector expansion `f(\mathbf{x}) = \sum_{i=1}^{\ell} y_i \alpha_i \phi(\mathbf{x}) \cdot \phi(\mathbf{x}_i) + b` (Section 4). `Constructed` state is inferred from learned support-vector representation `\mathbf{w} = \sum_{i=1}^{\ell} y_i \alpha_i \phi(\mathbf{x}_i)` (Section 4). `0D` output and `Fixed` output dynamics are inferred from binary label decisions `y_i \in \{-1,1\}` (Section 2.1).

### Task: Multiclass handwritten digit classification (one-vs-all)
- "we conduct experiments with the real-life problem of digit recognition." (Section 6)
- "The resolution of the database is  $16 \times 16$  pixels" and "The resolution of these patterns is  $28 \times 28$" (Section 6.2)
- "In all our experiments ten separators, one for each class, are constructed." and "Classification of an unknown patterns is done according to the maximum output of these ten classifiers." (Section 6.2)
- Inference: `Static` attention and `Constructed` state are inferred from the same fixed support-vector decision form and learned support-vector expansion defined in Section 4. `0D` output is inferred because the final prediction is a single class choice from the ten classifier outputs (Section 6.2). `Fixed` input/output dynamics are inferred from fixed image resolutions and a fixed set of ten class-specific separators (Section 6.2).
