# Scaling Instructable Agents Across Many Simulated Worlds (Not specified in the paper.)
Source: Scaling Instructable Agents Across Many Simulated Worlds (SIMA).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language-conditioned embodied control (instruction following) | image observations; language instructions | 1D (t); 3D (x, y, t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed | keyboard-and-mouse actions | 1D (t) (inferred) | Open (inferred) |
| Goal completion prediction | state representation from visual observations and language instructions (inferred) | 1D (t); 3D (x, y, t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | goal completion signal | 0D (inferred) | Fixed (inferred) |

## Summary
The paper centers on language-conditioned embodied control in interactive 3D environments, mapping multimodal inputs (visual observations plus natural-language instructions) to keyboard-and-mouse action streams. It also explicitly adds an auxiliary prediction task for goal completion during training. From the OCR text, the justified coverage spans 1D language plus spatiotemporal visual input (inferred as 3D (x, y, t)), with open interaction dynamics for control and a fixed scalar-style output for goal-completion prediction. The architecture description supports dynamic attention over memory and a constructed internal state representation.

## Evidence
### Task: Language-conditioned embodied control (instruction following)
- "the inputs are image observations and language instructions and the outputs are keyboard-and-mouse actions." (Section opening summary, before Section 1)
- "The SIMA agent maps visual observations and language instructions to keyboard-and-mouse actions (Figure 4)." (Section 3.3)
- "our agent (Figure 4) utilizes trained-from-scratch transformers that cross-attend to the different pretrained vision components, the encoded language instruction, and a Transformer-XL (Dai et al., 2019) that attends to past memory states to construct a state representation." (Section 3.3)
- Inference: `In Dimension = 1D (t); 3D (x, y, t)`, `In/Out Dynamics = Open`, `Attention Dynamic = Dynamic`, and `Out Dimension = 1D (t)` are inferred from the stated language-plus-image interface and real-time asynchronous interaction, supported by "Our agents interact with environments in real-time using a generic, human-like interface: the inputs are image observations and language instructions and the outputs are keyboard-and-mouse actions." (Section opening summary, before Section 1).

### Task: Goal completion prediction
- "We train this agent with behavioral cloning, as well as an auxiliary objective of predicting goal completion." (Section 3.3)
- "The resulting state representation is provided as input to a policy network that produces keyboard-and-mouse actions for sequences of 8 actions." (Section 3.3)
- Inference: the goal-completion task is treated as a scalar decision output (`Out Dimension = 0D`, `Out Dynamics = Fixed`), with input structure/dynamics and constructed state inherited from the same language-conditioned, temporally evolving state pathway used by the control model.
