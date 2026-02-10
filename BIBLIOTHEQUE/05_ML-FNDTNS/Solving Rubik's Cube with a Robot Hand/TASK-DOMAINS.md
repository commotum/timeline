# Solving Rubik's Cube with a Robot Hand (2019)
Source: Solving Rubik's Cube with a Robot Hand.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Block reorientation manipulation/control | Robot proprioceptive observations, block pose, and goal orientation observations | 4D (x, y, z, t) (inferred) | Capped | Static (inferred) | Constructed | Joint action commands to reorient the block | 4D (x, y, z, t) (inferred) | Capped |
| Rubik's cube manipulation/control | Robot proprioceptive observations plus Rubik's cube pose and six face-angle state estimates | 4D (x, y, z, t) (inferred) | Capped | Static (inferred) | Constructed | Joint action commands to execute flip/rotation subgoals | 4D (x, y, z, t) (inferred) | Capped |
| Block state estimation from vision | Three RGB camera images | 2D (x, y) (inferred) | Fixed | Static (inferred) | Direct (inferred) | Block position and orientation estimates | 3D (x, y, z) or (x, y, t); 0D (inferred) | Fixed |
| Rubik's cube state estimation from vision | Three RGB camera images | 2D (x, y) (inferred) | Fixed | Static (inferred) | Constructed (inferred) | Cube position/orientation and face-angle state estimates | 3D (x, y, z) or (x, y, t); 0D (inferred) | Fixed |

## Summary
The paper covers both manipulation/control and vision-based state estimation, on two embodied tasks: block reorientation and Rubik's cube solving. Vision tasks are supported by fixed multi-camera 2D image inputs and fixed-size state outputs, while control tasks run as capped episodes with recurrent policies. The strongest dimension spread justified by the text is from 2D (x, y) visual inputs to 4D (x, y, z, t) embodied manipulation interaction over time. Attention is static across the described architectures, while state is constructed for recurrent control and Rubik face-angle tracking, and direct for block pose estimation.

## Evidence
### Task: Block reorientation manipulation/control
- "The goal of the block reorientation task is to rotate a block into a desired goal orientation." (Section 2.1)
- "Actions are relative changes in generalized joint position coordinates." (Section 6.1)
- "Time out limits are 400 timesteps for block reorientation and 800 timesteps<sup>10</sup> for the Rubik's Cube." (Section 6.1)
- "The policy is still recurrent since only a policy with access to some form of memory can perform meta-learning. We still use a single feed-forward layer with a ReLU activation [72] followed by a single LSTM layer [45]." (Section 6.2)
- Inference: 4D (x, y, z, t) is inferred from time-indexed in-hand manipulation/control of a 3D object; Static attention is inferred from fixed observation fields fed through a fixed policy architecture; Capped dynamics is supported by explicit episode timeout limits.

### Task: Rubik's cube manipulation/control
- "In this work, the key problem is thus about sensing and control, *not* finding the solution sequence. More concretely, we need to obtain the state of the Rubik's cube (i.e. its pose as well as its 6 face angles) and use that information to control the robot hand such that each subgoal is successfully achieved." (Section 2.2)
- "We consider two types of *subgoals*: A *rotation* corresponds to rotating a single face of the Rubik's cube by 90 degrees in the clockwise or counter-clockwise direction. A *flip* corresponds to moving a different face of the Rubik's cube to the top." (Section 2.2)
- "Time out limits are 400 timesteps for block reorientation and 800 timesteps<sup>10</sup> for the Rubik's Cube." (Section 6.1)
- "The policy is still recurrent since only a policy with access to some form of memory can perform meta-learning." (Section 6.2)
- Inference: 4D (x, y, z, t) is inferred because the policy performs embodied manipulation trajectories over time from spatial state estimates; Static attention is inferred from fixed policy inputs/architecture; Capped dynamics is supported by explicit timeout-based episode limits.

### Task: Block state estimation from vision
- "We train ADR-enhanced vision models to do state estimation for both the block reorientation [77] and Rubik's cube task." (Section 8.3)
- "Table 4: Performance of vision models at different ADR entropy levels for the block reorientation state estimation task." (Section 8.3)
- "We still use the same 3 RGB Basler cameras for vision pose estimation." (Section 3.1)
- Inference: 2D (x, y) input is inferred from camera images; output dimension includes 3D (position) and 0D components (orientation parameters); Static attention and Direct state are inferred because the described vision predictor is feed-forward on fixed image inputs without explicit temporal memory for this block task.

### Task: Rubik's cube state estimation from vision
- "As in [77], the control policy described in Section 6 receives object state estimates from a vision system consisting of three cameras and a neural network predictor. In this work, the policy requires estimates for all six face angles in addition to the position and orientation of the cube." (Section 7)
- "Our vision model has a similar setup as in [77], taking as input an image from each of three RGB Basler cameras located at the left, right, and top of the cage (see Figure 4(a))." (Section 7.1)
- "These three feature maps are then flattened, concatenated, and fed into a stack of fully-connected layers which ultimately produce predictions sufficient for tracking the full state of the cube, including the position, orientation, and face angles." (Section 7.1)
- "These decomposed face angle predictions are then fed into post-processing logic (See Appendix C Algorithm 5) to track the rotation of all face angles, which are in turn passed along to the policy." (Section 7.1)
- Inference: 2D (x, y) input is inferred from RGB frames; output dimension mixes 3D pose and 0D angular/class outputs; Constructed state is inferred from explicit temporal tracking/post-processing of face-angle state.
