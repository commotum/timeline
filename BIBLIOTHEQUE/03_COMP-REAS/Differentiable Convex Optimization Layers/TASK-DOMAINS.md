# Differentiable Convex Optimization Layers (Not specified in the paper)
Source: Differentiable Convex Optimization Layers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classification model fitting (logistic regression) | Training data (feature vectors x_i, labels y_i) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Model parameters \\theta* | 1D (t) (inferred) | Not specified in the paper. |
| Stochastic control (policy optimization) | State x_t (and disturbances \\omega_t) | 1D (t) (inferred) | Open (inferred) | Not specified in the paper. | Not specified in the paper. | Control action u = \\phi(x_t) | 1D (t) (inferred) | Open (inferred) |
| Vector projection/normalization (activation/attention-style layers) | Vector x (and optional constraints u, k) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Projected/normalized vector y | 1D (t) (inferred) | Not specified in the paper. |
| Quadratic program solving (OptNet QP layer) | QP problem data (Q, q, A, b, G, h) | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | QP solution vector x | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper's concrete applications are classification model fitting via logistic regression and stochastic control policy optimization, and it also provides additional examples of vector projection/normalization layers and a generic QP layer. Inputs and outputs are described primarily as vectors (and, for QP inputs, matrices), corresponding to 1D (and 2D) address spaces. Only the stochastic control problem explicitly uses an open-ended time horizon; other dynamics are not specified. Attention and state dynamics are not described for these tasks.

## Evidence
### Task: Classification model fitting (logistic regression)
- "We are given training data  $(x_i, y_i)_{i=1}^N$" (Section 6.1 Data poisoning attack)
- "$x_i \in \mathbf{R}^n$  are feature vectors and  $y_i \in \{0, 1\}$  are the labels." (Section 6.1 Data poisoning attack)
- "fit a model for this classification problem by solving" (Section 6.1 Data poisoning attack)
- "Let  $\theta^*$  be optimal for (6)." (Section 6.1 Data poisoning attack)
- Inference: Marked input/output dimension as 1D because the data and parameters are vectors ($x_i \in \mathbf{R}^n$, $\theta$ in (6)). (Section 6.1 Data poisoning attack)

### Task: Stochastic control (policy optimization)
- "We consider a stochastic control problem of the form" (Section 6.2 Convex approximate dynamic programming)
- "$x_{t+1} = Ax_t + B\phi(x_t) + \omega_t, \quad t = 0, 1, \dots,$" (Section 6.2 Convex approximate dynamic programming)
- "$x_t \in \mathbf{R}^n$  is the state" (Section 6.2 Convex approximate dynamic programming)
- " $\phi : \mathbf{R}^n \to \mathcal{U} \subseteq \mathbf{R}^m$  is the policy" (Section 6.2 Convex approximate dynamic programming)
- "Evaluating  $\phi$  corresponds to solving the SOCP" (Section 6.2 Convex approximate dynamic programming)
- "with variable u and parameters P, Q, q, and  $x_t$." (Section 6.2 Convex approximate dynamic programming)
- Inference: Used 1D for input/output because state and control are vectors ($x_t \in \mathbf{R}^n$, policy output in $\mathbf{R}^m$). Used Open dynamics because the problem is indexed for t = 0, 1, ... with an infinite-horizon objective. (Section 6.2 Convex approximate dynamic programming)

### Task: Vector projection/normalization (activation/attention-style layers)
- "We present the implementation of common neural networks layers" (Appendix E Additional examples)
- "In the below problems, the optimization variable is y (unless stated otherwise)." (Appendix E Additional examples)
- "can be interpreted as projecting a point  $x \in \mathbf{R}^n$  onto the non-negative orthant" (Appendix E Additional examples)
- "can be interpreted as projecting a point  $x \in \mathbf{R}^n$  onto the interior of the (n-1)-simplex" (Appendix E Additional examples)
- Inference: Marked input/output dimension as 1D because these layers map vectors $x \in \mathbf{R}^n$ to vector outputs y. (Appendix E Additional examples)

### Task: Quadratic program solving (OptNet QP layer)
- "The OptNet layer is a solution to a convex quadratic program of the form" (Appendix E Additional examples)
- "where  $x \in \mathbf{R}^n$  is the optimization variable" (Appendix E Additional examples)
- "the problem data are  $Q \in \mathbf{R}^{n \times n}$ ,  $q \in \mathbf{R}^n$ ,  $A \in \mathbf{R}^{m \times n}$" (Appendix E Additional examples)
- Inference: Treated inputs as 2D/1D because the problem data include matrices and vectors (Q, q, A, etc.), and output as 1D because $x \in \mathbf{R}^n$. (Appendix E Additional examples)
