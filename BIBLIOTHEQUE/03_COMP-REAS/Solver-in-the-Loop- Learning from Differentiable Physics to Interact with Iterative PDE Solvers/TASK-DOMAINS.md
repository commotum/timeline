# Solver-in-the-Loop: Learning from Differentiable Physics to Interact with Iterative PDE-Solvers (Not specified in the paper)
Source: Solver-in-the-Loop- Learning from Differentiable Physics to Interact with Iterative PDE Solvers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Numerical-error correction for iterative PDE solver rollouts | PDE phase-space states from the source manifold (e.g., velocity and coupled advection-diffusion fields) | 2D (x, y); 3D (x, y, z) or (x, y, t) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Additive correction field applied at solver iterations | 2D (x, y); 3D (x, y, z) or (x, y, t) | Not specified in the paper. |
| Initial-guess prediction for conjugate-gradient Poisson solving | Poisson-problem field input from velocity divergence terms (∇·u) | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Initial pressure-field guess for CG iterations | 2D (x, y) (inferred) | Not specified in the paper. |

## Summary
The paper covers two solver-interaction prediction tasks: learning additive correction fields for iterative PDE rollouts and learning initial guesses for iterative CG solving of Poisson systems. The explicitly stated spatial regimes span 2D and 3D fluid settings; the CG row is grid-based but its exact dimensionality is inferred from context. Dynamics constraints (Fixed/Capped/Open) are not explicitly specified in the OCR text for either task interface. The model behavior is best supported as Static attention and Direct state inferences, based on a fully convolutional correction network applied to current solver states.

## Evidence
### Task: Numerical-error correction for iterative PDE solver rollouts
- "We target the problem of reducing numerical errors of iterative PDE solvers and compare different learning approaches for finding complex correction functions." (Section Abstract)
- "Our learning goal is to arrive at a correction operator  $\mathcal{C}(s)$  such that a solution to which the correction is applied has a lower error than an unmodified solution... The correction function  $\mathcal{C}(s|\theta)$  is represented as a deep neural network with weights  $\theta$  and receives the state s to infer an additive correction field with the same dimension." (Section 2 Learning to Reduce Numerical Errors)
- "In total, we target four scenarios: pure non-linear advection-diffusion (Burger's equation), two-dimensional Navier-Stokes flow, Navier-Stokes coupled with a second advection-diffusion equation for a buoyancy-driven flow, and a 3D Navier-Stokes case." (Section 3.1 Model Equations and Data Generation)
- Inference: Attention Dynamic = Static (inferred) and State Dynamic = Direct (inferred) are based on "The neural network component  $F(s \mid \theta)$  of the correction function is realized with a fully convolutional architecture," which describes direct mapping from current state fields without explicit runtime input-selection or external memory construction (Section 3.2 Training Procedure).

### Task: Initial-guess prediction for conjugate-gradient Poisson solving
- "On the side of implicit solvers, we consider the Poisson problem [37], which is an essential component of many PDE models. Here, our method outperforms existing techniques on predicting initial guesses for a conjugate gradient (CG) solver by receiving feedback from the solver at training time." (Section 1 Introduction)
- "As our learning objective, we target the inference of initial guesses for CG solvers [22]. Following previous work [57], we target Poisson problems of the form  $\nabla \cdot \nabla p = \nabla \cdot \boldsymbol{u}$ ..." (Section 4 Results, Conjugate Gradient Solver)
- "to produce the pressure field p, we instead target the learning objective to produce an initial guess, which is improved by a regular CG solver until a given accuracy threshold is reached." (Section 4 Results, Conjugate Gradient Solver)
- Inference: In Dimension = 2D (x, y) (inferred), Attention Dynamic = Static (inferred), and State Dynamic = Direct (inferred) are supported by grid-based wording ("for each grid cell") in the CG discussion and reuse of ANN-based field mapping without described dynamic retrieval/memory mechanisms (Section 4 Results; Section 3.2 Training Procedure).
