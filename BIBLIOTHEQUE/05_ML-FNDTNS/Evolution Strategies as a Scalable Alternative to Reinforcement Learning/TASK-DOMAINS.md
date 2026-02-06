# Evolution Strategies as a Scalable Alternative to Reinforcement Learning (Not specified in the paper)
Source: Evolution Strategies as a Scalable Alternative to Reinforcement Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control (MuJoCo robotic control) | input observations | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | actions | Not specified in the paper. | Not specified in the paper. |
| Control (Atari game playing) | raw pixel input | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | actions | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper covers two task domains: robotic control in MuJoCo and Atari 2600 game playing. Inputs include simulator observations and raw pixel inputs, while outputs are actions for control. Beyond the pixel modality, the paper does not specify dynamics, attention, or state properties, so most fields remain unspecified and the Atari input dimension is inferred as 2D imagery.

## Evidence
### Task: Control (MuJoCo robotic control)
- "controlling robots in the MuJoCo physics simulator" (Section 1 Introduction)
- "continuous robotic control problems in the OpenAI Gym" (Section 4.1 Mu.JoCo)
- "mapping to continuous actions" (Section 4.1 Mu.JoCo)
- "input observations" (Section 2.2 The impact of network parameterization)

### Task: Control (Atari game playing)
- "playing Atari games with pixel inputs" (Section 1 Introduction)
- "on 51 Atari 2600 games available in OpenAI Gym" (Section 4.2 Atari)
- "trained on raw pixel input" (Table 2)
- Inference: In Dimension is 2D (x, y) because the paper specifies "pixel inputs" and "raw pixel input," indicating image grids.
