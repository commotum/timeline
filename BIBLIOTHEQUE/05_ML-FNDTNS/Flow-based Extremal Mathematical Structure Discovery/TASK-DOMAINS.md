# Flow-Based Extremal Mathematical Structure Discovery (2026)
Source: Flow-based Extremal Mathematical Structure Discovery.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Optimization (sphere packing in hypercube) | Sphere-center configurations (point sets) in a d-dimensional unit hypercube | 3D (x, y, z) (inferred) | Fixed | Not specified in the paper. | Not specified in the paper. | Optimized sphere-center configurations (non-overlapping spheres) | 3D (x, y, z) (inferred) | Fixed |
| Optimization (Heilbronn triangle problem) | Point sets in the unit square | 2D (x, y) | Fixed | Not specified in the paper. | Not specified in the paper. | Point sets maximizing minimum triangle area | 2D (x, y) | Fixed |
| Optimization (circle packing, max sum of radii) | Circle configurations (centers and radii) in the unit square | 2D (x, y) | Fixed | Not specified in the paper. | Not specified in the paper. | Circle configurations maximizing sum of radii | 2D (x, y) | Fixed |
| Optimization (star discrepancy minimization) | Point sets in [0,1]^2 | 2D (x, y) | Fixed | Not specified in the paper. | Not specified in the paper. | Low star-discrepancy point sets | 2D (x, y) | Fixed |

## Summary
The paper applies FLowBoost to four geometric optimization tasks: sphere packing in a d-dimensional hypercube and three unit-square point-set problems (Heilbronn triangle, circle packing with maximal sum of radii, and star discrepancy minimization). The explicitly described tasks operate on fixed-size configurations, with 2D spatial point/circle layouts for three tasks and d-dimensional sphere-center configurations for sphere packing (mapped to the 3D label due to the glossary's limits). Attention and state dynamics are not specified in the paper for these tasks.

## Evidence
### Task: Optimization (sphere packing in hypercube)
- "Sphere Packing. We evaluate FlowBoost on the classical problem of packing N non-overlapping spheres of radius r inside a d-dimensional unit hypercube" (Section 3.1 Overview)
- "operating on the point-cloud x=(x_1,\ldots,x_N)\in(\mathbb{R}^d)^N, where each configuration x\in\mathbb{R}^{d\times N} is treated as a set of N tokens with d-dimensional coordinates" (Section 3.2 Sphere Packing in Hypercube)
- Inference: Labeled the dimension as 3D (x, y, z) because the task is defined in a d-dimensional hypercube (including d=3 and d=12), while the glossary provides no label beyond 4D. (Section 3.1 Overview)

### Task: Optimization (Heilbronn triangle problem)
- "The Heilbronn Problem. We study the classical Heilbronn triangle problem in the unit square: for a point set X = {p_1, \dots, p_n} \subset [0, 1]^2" (Section 3.1 Overview)
- "the goal is to maximize A_{min}(X) over all n-point configurations" (Section 3.1 Overview)

### Task: Optimization (circle packing, max sum of radii)
- "Circle packing with maximal sum of radii. We study circle packings in the unit square where the objective is to maximize the sum of radii for a fixed number N of circles" (Section 3.1 Overview)
- "A configuration consists of centers and radii X = ((p_1, r_1), \dots, (p_n, r_n)), \qquad p_i = (x_i, y_i) \in [0, 1]^2, \quad r_i \ge 0." (Section 3.4 Circles in Unit Square with Maximal Sum of Radii)

### Task: Optimization (star discrepancy minimization)
- "Star discrepancy problem. We also apply FLowBoost to constructing low star-discrepancy point sets in [0, 1]^2" (Section 3.1 Overview)
- "For P = {p_1, \ldots, p_N} \subset [0, 1]^2, the (anchored) star discrepancy is the minimax quantity" (Section 3.1 Overview)
