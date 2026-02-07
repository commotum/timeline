# An implicit factorized transformer with applications to fast prediction of three-dimensional turbulence (Not specified in the paper.)
Source: Implicit Factorized Transformer (IFactFormer).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prediction (forecasting) of turbulence velocity fields | velocity fields from preceding time points | 4D (x, y, z, t) | Fixed | Static (inferred) | Constructed (inferred) | velocity field at subsequent time point | 3D (x, y, z) | Fixed |

## Summary
The paper defines a single task: forecasting three-dimensional turbulence by predicting future velocity fields from several previous time steps. Inputs are spatiotemporal 3D velocity fields (4D with time), and outputs are 3D velocity fields at the next step, both with fixed grid sizes and fixed input history length. The attention is described as standard self-attention over fixed inputs and the model iterates a latent field, so the Attention and State dynamics are inferred as Static and Constructed, respectively.

## Evidence
### Task: prediction (forecasting) of turbulence velocity fields
- "The task of the IFactFormer is to adopt the velocity fields from the preceding several time-nodes to predict the velocity fields of the subsequent time-nodes." (Section 3.3)
- "velocity fields from the five preceding time points (U1, U2, U3, U4, U5) are used to predict the velocity field" (Section 4)
- "the filtered direct numerical simulation (fDNS) data, with dimensions of [45 x 600 x 32 x 32 x 32 x 3]" (Section 4)
- "assuming that the input size are (N,T), where N signifies the quantity of grid points and T corresponds to the frame count" (Section 3.3)
- Inference: Attention is Static because self-attention is computed from the fixed input vectors without runtime selection ("In self-attention mechanisms, all of them are calculated from the same inputs vector u_i" in Section 3.2).
- Inference: State is Constructed because the model maintains and updates an internal latent field across iterations ("the single field subject to iterative updates v(x,0Δs)→v(x,1Δs)→…→v(x,LΔs) via implicit factorized attention layers" in Section 3.3).
