# REINFORCEMENT LEARNING NEURAL TURING MACHINES - REVISED (Not specified in the paper)
Source: Reinforcement Learning Neural Turing Machines - Revised.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Copy | symbol sequence on input tape | 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | symbol sequence copied to output tape | 1D (t) | Not specified in the paper. |
| DuplicatedInput | triplicated symbol sequence on input tape | 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | symbol sequence with every third input symbol | 1D (t) | Not specified in the paper. |
| Reverse | symbol sequence on input tape | 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | reversed symbol sequence | 1D (t) | Not specified in the paper. |
| RepeatCopy | m plus symbol sequence on input tape | 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | symbol sequence repeated m times (m in {2,3}) | 1D (t) | Not specified in the paper. |
| ForwardReverse | symbol sequence on input tape (forward-only head) | 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | reversed symbol sequence | 1D (t) | Not specified in the paper. |
| Sorting | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Long integer addition (base 3) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper evaluates 1D symbol-sequence transduction tasks on input/output tapes: copy, duplicated-input filtering, reverse, repeat-copy, and forward-only reverse. The model makes discrete tape-access decisions and writes memory values, so attention is dynamic and state is constructed (inferred). It also mentions sorting and long integer addition (base 3) as additional tasks, but does not specify their input/output domains or dynamics.

## Evidence
### Task: Copy
- "A generic input is  $x_1x_2x_3...x_C\varnothing$  and the desired output is  $x_1x_2...x_C\varnothing$ ." (Section 6 Tasks)
- "The RL—NTM makes discrete decisions regarding the move over the input tape, the memory tape, and whether to make a prediction at a given timestep." (Figure 1 caption)
- "The memory value vector is a vector of content that is stored in the memory cell." (Figure 1 caption)
- Inference: Marked Attention as Dynamic and State as Constructed based on the discrete tape decisions and stored memory value vector described in Figure 1.

### Task: DuplicatedInput
- "Thus each input symbol is replicated three times, so the RL-NTM must emit every third input symbol." (Section 6 Tasks)
- "The RL—NTM makes discrete decisions regarding the move over the input tape, the memory tape, and whether to make a prediction at a given timestep." (Figure 1 caption)
- "The memory value vector is a vector of content that is stored in the memory cell." (Figure 1 caption)
- Inference: Marked Attention as Dynamic and State as Constructed based on the discrete tape decisions and stored memory value vector described in Figure 1.

### Task: Reverse
- "A generic input is  $x_1x_2 \dots x_{C-1}x_C\varnothing$  and the desired output is  $x_Cx_{C-1}\dots x_2x_1\varnothing$ ." (Section 6 Tasks)
- "The RL—NTM makes discrete decisions regarding the move over the input tape, the memory tape, and whether to make a prediction at a given timestep." (Figure 1 caption)
- "The memory value vector is a vector of content that is stored in the memory cell." (Figure 1 caption)
- Inference: Marked Attention as Dynamic and State as Constructed based on the discrete tape decisions and stored memory value vector described in Figure 1.

### Task: RepeatCopy
- "Thus the goal is to copy the input m times, where m can be only 2 or 3." (Section 6 Tasks)
- "The RL—NTM makes discrete decisions regarding the move over the input tape, the memory tape, and whether to make a prediction at a given timestep." (Figure 1 caption)
- "The memory value vector is a vector of content that is stored in the memory cell." (Figure 1 caption)
- Inference: Marked Attention as Dynamic and State as Constructed based on the discrete tape decisions and stored memory value vector described in Figure 1.

### Task: ForwardReverse
- "The task is identical to Reverse, but the RL-NTM is only allowed to move its input tape pointer forward." (Section 6 Tasks)
- "The RL—NTM makes discrete decisions regarding the move over the input tape, the memory tape, and whether to make a prediction at a given timestep." (Figure 1 caption)
- "The memory value vector is a vector of content that is stored in the memory cell." (Figure 1 caption)
- Inference: Marked Attention as Dynamic and State as Constructed based on the discrete tape decisions and stored memory value vector described in Figure 1.

### Task: Sorting
- "Tasks we found to be too difficult include sorting and long integer addition (in base 3 for simplicity)" (Section 9 Experiments)
- "The RL—NTM makes discrete decisions regarding the move over the input tape, the memory tape, and whether to make a prediction at a given timestep." (Figure 1 caption)
- "The memory value vector is a vector of content that is stored in the memory cell." (Figure 1 caption)
- Inference: Marked Attention as Dynamic and State as Constructed based on the discrete tape decisions and stored memory value vector described in Figure 1.

### Task: Long integer addition (base 3)
- "Tasks we found to be too difficult include sorting and long integer addition (in base 3 for simplicity)" (Section 9 Experiments)
- "The RL—NTM makes discrete decisions regarding the move over the input tape, the memory tape, and whether to make a prediction at a given timestep." (Figure 1 caption)
- "The memory value vector is a vector of content that is stored in the memory cell." (Figure 1 caption)
- Inference: Marked Attention as Dynamic and State as Constructed based on the discrete tape decisions and stored memory value vector described in Figure 1.
