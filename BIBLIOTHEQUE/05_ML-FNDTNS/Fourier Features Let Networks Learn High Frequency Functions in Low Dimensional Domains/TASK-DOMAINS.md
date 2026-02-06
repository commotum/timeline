# Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains (Not specified in the paper.)
Source: Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1D function regression | 1D coordinates on [0,1) (inferred) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | signal value (inferred) | 0D (inferred) | Fixed (inferred) |
| 2D image regression | 2D pixel coordinate | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | RGB value of an image | 0D (inferred) | Fixed (inferred) |
| 3D shape regression (occupancy) | 3D point coordinates (inferred) | 3D (x, y, z) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | occupancy label (0/1 inside/outside) | 0D (inferred) | Fixed (inferred) |
| 2D CT density regression (indirect supervision) | 2D pixel coordinate | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | volume density at that location | 0D (inferred) | Fixed (inferred) |
| 3D MRI response regression (indirect supervision) | 3D voxel coordinate | 3D (x, y, z) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | response at that location | 0D (inferred) | Fixed (inferred) |
| 3D inverse rendering for view synthesis (indirect supervision) | 3D location | 3D (x, y, z) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | color and volume density | 0D (inferred) | Fixed (inferred) |

## Summary
The paper's experiments cover coordinate-based regression tasks from 1D signals to 2D images and 3D fields, with inputs as low-dimensional coordinates and outputs as per-coordinate values. Tasks include direct supervision (1D signal, 2D image, 3D occupancy) and indirect supervision through forward models (2D CT, 3D MRI, and 3D view synthesis). Across the tasks, the interface is a fixed-size coordinate input with fixed-size value outputs, and the MLP operates with static attention and direct (non-persistent) state, as inferred from the task descriptions.

## Evidence
### Task: 1D function regression
- "we investigate the effects of the Fourier feature mapping in the setting of 1D function regression." (Section 5)
- "We train MLPs to learn signals f defined on the interval [0,1)." (Section 5)
- Inference: The quoted description of signals on [0,1) implies 1D coordinate inputs and scalar signal outputs; dimensions, fixed dynamics, static attention, and direct state are inferred from the coordinate-based MLP setup.

### Task: 2D image regression
- "In this task, we train an MLP to regress from a 2D input pixel coordinate to the corresponding RGB value of an image." (Section 6.2)
- Inference: "2D input pixel coordinate" supports 2D (x, y) input dimension and fixed dynamics; per-coordinate RGB output implies 0D output dimension with fixed dynamics, static attention, and direct state.

### Task: 3D shape regression (occupancy)
- "which is trained to output 0 for points outside the shape and 1 for points inside the shape." (Section 6.2)
- Inference: The task is 3D shape regression with point-based occupancy labels, implying 3D coordinate inputs, 0D outputs, fixed dynamics, static attention, and direct state.

### Task: 2D CT density regression (indirect supervision)
- "we train an MLP that takes in a 2D pixel coordinate and predicts the corresponding volume density at that location." (Section 6.2)
- "The network is indirectly supervised by the loss between a sparse set of ground-truth integral projections" (Section 6.2)
- Inference: The 2D coordinate input implies 2D (x, y) input dimension and fixed dynamics; per-coordinate density output implies 0D output dimension with fixed dynamics, static attention, and direct state.

### Task: 3D MRI response regression (indirect supervision)
- "we train an MLP that takes in a 3D voxel coordinate and predicts the corresponding response at that location." (Section 6.2)
- "The network is indirectly supervised by the loss between a sparse set of ground-truth Fourier transform coefficients" (Section 6.2)
- Inference: The 3D voxel coordinate input implies 3D (x, y, z) input dimension and fixed dynamics; per-coordinate response output implies 0D output dimension with fixed dynamics, static attention, and direct state.

### Task: 3D inverse rendering for view synthesis (indirect supervision)
- "we train a coordinate-based MLP that takes in a 3D location and outputs a color and volume density." (Section 6.2)
- "This MLP is indirectly supervised by the loss between the set of 2D image observations" (Section 6.2)
- Inference: The 3D location input implies 3D (x, y, z) input dimension and fixed dynamics; per-coordinate color/density output implies 0D output dimension with fixed dynamics, static attention, and direct state.
