# On the difficulty of training recurrent neural networks (2013)
Source: On the difficulty of training recurrent neural networks (Pascanu, Mikolov & Bengio).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification (temporal order problem) | discrete symbol sequence | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Constructed (inferred) | order class (AA, AB, BA, BB) | 0D (inferred) | Fixed (inferred) |
| addition problem | Not specified in the paper. | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Constructed (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| multiplication problem | Not specified in the paper. | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Constructed (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| 3-bit temporal order problem | Not specified in the paper. | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Constructed (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| random permutation problem | Not specified in the paper. | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Constructed (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| noiseless memorization problem | Not specified in the paper. | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Constructed (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| prediction (polyphonic music) | polyphonic music sequences (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | polyphonic music events per time step (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| prediction (character-level language modeling) | character sequence (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | next character per time step (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| prediction (character-level, 5-step ahead) | character sequence (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | 5th character in the future | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper evaluates RNNs on multiple sequence-focused tasks: a temporal order classification problem, several named pathological sequence benchmarks (addition, multiplication, permutation, memorization), and natural tasks in polyphonic music prediction and character-level language modeling (including a 5-step-ahead variant). Where the paper is explicit, inputs and outputs are temporal sequences and labels, supporting 1D (t) domains and capped sequence lengths for the synthetic benchmarks; many other input/output details are not specified. Attention dynamics are not described, while state is inferred as constructed from the RNN formulation.

## Evidence
### Task: classification (temporal order problem)
- "The input is a long stream of discrete symbols." (Section 4.1.1. The temporal order problem)
- "The task consists in classifying the order (either AA, AB, BA, BB) at the end of the sequence." (Section 4.1.1. The temporal order problem)
- Inference: Marked 1D (t) and Capped because the task is a "long stream" and is evaluated on "any sequence of length 50 up to 200" (Section 4.1.1). Marked Constructed due to "input  $\mathbf{u}_t$  and state  $\mathbf{x}_t$  for time step t" (Section 1.1). Marked 0D/Fixed output because it is a single class among "either AA, AB, BA, BB" (Section 4.1.1).

### Task: addition problem
- "the addition problem, the multiplication problem, the 3-bit temporal order problem, the random permutation problem and the noiseless memorization problem" (Section 4.1.2. Other pathological tasks)
- Inference: Marked 1D (t) and Capped because "the first 4 problems" use "lengths up to 200" (Section 4.1.2). Marked Constructed due to "input  $\mathbf{u}_t$  and state  $\mathbf{x}_t$  for time step t" (Section 1.1).

### Task: multiplication problem
- "the addition problem, the multiplication problem, the 3-bit temporal order problem, the random permutation problem and the noiseless memorization problem" (Section 4.1.2. Other pathological tasks)
- Inference: Marked 1D (t) and Capped because "the first 4 problems" use "lengths up to 200" (Section 4.1.2). Marked Constructed due to "input  $\mathbf{u}_t$  and state  $\mathbf{x}_t$  for time step t" (Section 1.1).

### Task: 3-bit temporal order problem
- "the addition problem, the multiplication problem, the 3-bit temporal order problem, the random permutation problem and the noiseless memorization problem" (Section 4.1.2. Other pathological tasks)
- Inference: Marked 1D (t) and Capped because "the first 4 problems" use "lengths up to 200" (Section 4.1.2). Marked Constructed due to "input  $\mathbf{u}_t$  and state  $\mathbf{x}_t$  for time step t" (Section 1.1).

### Task: random permutation problem
- "the addition problem, the multiplication problem, the 3-bit temporal order problem, the random permutation problem and the noiseless memorization problem" (Section 4.1.2. Other pathological tasks)
- Inference: Marked 1D (t) and Capped because "the first 4 problems" use "lengths up to 200" (Section 4.1.2). Marked Constructed due to "input  $\mathbf{u}_t$  and state  $\mathbf{x}_t$  for time step t" (Section 1.1).

### Task: noiseless memorization problem
- "the addition problem, the multiplication problem, the 3-bit temporal order problem, the random permutation problem and the noiseless memorization problem" (Section 4.1.2. Other pathological tasks)
- Inference: Marked 1D (t) and Capped because it uses "sequence length (50, 100, 150 and 200)" (Section 4.1.2). Marked Constructed due to "input  $\mathbf{u}_t$  and state  $\mathbf{x}_t$  for time step t" (Section 1.1).

### Task: prediction (polyphonic music)
- "We address the task of polyphonic music prediction" (Section 4.2. Natural problems)
- "negative log likelihood per time step" (Table 1)
- Inference: Marked sequence input/output and 1D (t) because the task is "polyphonic music prediction" with results reported "per time step" (Section 4.2; Table 1). Marked Constructed due to "input  $\mathbf{u}_t$  and state  $\mathbf{x}_t$  for time step t" (Section 1.1).

### Task: prediction (character-level language modeling)
- "language modelling at the character level on the Penn Treebank dataset" (Section 4.2. Natural problems)
- "predict the 5th character in the future (instead of the next)." (Section 4.2. Natural problems)
- Inference: Marked character-sequence input/output and 1D (t) because it is "character level" modeling and the base task predicts "the next" character (Section 4.2). Marked Constructed due to "input  $\mathbf{u}_t$  and state  $\mathbf{x}_t$  for time step t" (Section 1.1).

### Task: prediction (character-level, 5-step ahead)
- "We also explore a modified version of the task, where we require to predict the 5th character in the future (instead of the next)." (Section 4.2. Natural problems)
- "language modelling at the character level on the Penn Treebank dataset" (Section 4.2. Natural problems)
- Inference: Marked character-sequence input and 1D (t) because it is "character level" modeling with future-character prediction (Section 4.2). Marked Constructed due to "input  $\mathbf{u}_t$  and state  $\mathbf{x}_t$  for time step t" (Section 1.1).
