# Relational recurrent neural networks (2018)
Source: Relational recurrent neural networks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Comparative reasoning (N-th farthest vector selection) | Sequence of sampled vectors with labels and query variables (n, m) | 1D (t) | Fixed | Dynamic (inferred) | Constructed (inferred) | Selected answer to the farthest-vector query | 0D | Fixed |
| Program evaluation (Learning to Execute) | Sequence of pseudo-code characters | 1D (t) | Capped | Dynamic (inferred) | Constructed (inferred) | Numeric sequence of characters (program execution output) | 1D (t) | Capped |
| Sequence memorization/permutation (copy, reverse, double) | Sequence of symbols/tokens | 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Permuted or duplicated symbol sequence | 1D (t) | Not specified in the paper. |
| Partially observed reinforcement learning (Mini Pacman with viewport) | 5 x 5 viewport observations around the agent over time | 3D (x, y, t) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Navigation actions/policy decisions (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Relational planning/control in BoxWorld | 5 x 5 RGB viewport plus held-key indicator over time | 3D (x, y, t) (inferred) | Capped | Dynamic (inferred) | Constructed (inferred) | Agent actions (up, down, left, right) | 1D (t) (inferred) | Capped (inferred) |
| Word language modeling | Sequence of observed words w_<t | 1D (t) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | Next-word predictions / conditional word probabilities | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper covers supervised comparative reasoning, symbolic program execution, sequence memorization/permutation, partially observed reinforcement-learning control/planning, and word-level language modeling. Task inputs are primarily temporal sequences (1D (t)), with RL tasks additionally using spatial viewport observations over time (inferred 3D (x, y, t)). Dynamics include Fixed ($N^{th}$ Farthest), Capped (program evaluation and BoxWorld), and Open (inferred for language modeling), while some Mini Pacman and memorization bounds are not explicitly specified. Across tasks, the RMC evidence supports inferred Dynamic attention and Constructed state.

## Evidence
### Task: Comparative reasoning (N-th farthest vector selection)
- " $N^{th}$  Farthest The  $N^{th}$  Farthest task is designed to stress a capacity for relational reasoning across time. Inputs are a sequence of randomly sampled vectors, and targets are answers to a question of the form: \"What is the  $n^{th}$  farthest vector (in Euclidean distance) from vector m?\"" (Section 4.1 Illustrative supervised tasks)
- "Inputs consisted of sequences of eight randomly sampled, 16-dimensional vectors from a uniform distribution  $x_t \sim \mathcal{U}(-1,1)$ , and vector labels  $l_t \sim \{1,2,...,8\}$ , encoded as a one-hot vectors and sampled without replacement." (Section A.1 $N^{th}$ Farthest)
- Inference: "Dynamic" attention and "Constructed" state are inferred from the model description: "Using MHDPA, each memory will attend over all of the other memories, and will update its content based on the attended information." and "The output of this computation is a new memory where information is blended across memories based on their attention weights." and "An MLP is applied row-wise to the output of the MHDPA module (a), and the resultant memory matrix is gated, and passed on as the core output or next memory state." (Section 3.1; Figure 1 caption)

### Task: Program evaluation (Learning to Execute)
- "**Program Evaluation** The *Learning to Execute* (LTE) dataset [25] consists of algorithmic snippets from a Turing complete programming language of pseudo-code, and is broken down into three categories: *addition, control,* and *full program*." (Section 4.1 Illustrative supervised tasks)
- "Inputs are a sequence of characters over an alphanumeric vocabulary representing such snippets, and the target is a numeric sequence of characters that is the execution output for the given programmatic input." (Section 4.1 Illustrative supervised tasks)
- "The samples were parameterized by literal length and nesting depth which define the length of terminal values in the program snippets and the level of program operation nesting. Within each batch the literal length and nesting value was sampled uniformly up to the maximum value for each - this is consistent with the  $\mathit{Mix}$  curriculum strategy from [25]." (Section A.2 Program Evaluation)
- "It also worth noting that the modulus operation was applied to  $\mathit{addition}$ ,  $\mathit{control}$ , and  $\mathit{full}$   $\mathit{program}$  samples so as to bound the output to the maximum literal length in case of longer for-loops." (Section A.2 Program Evaluation)
- Inference: "Dynamic" attention and "Constructed" state are inferred from the shared RMC architecture used in these tasks (Sections 3.1-3.3).

### Task: Sequence memorization/permutation (copy, reverse, double)
- "To also assess model performance on classical sequence tasks we also evaluated on *memorization tasks*, in which the output is simply a permuted form of the input rather than an evaluation from a set of operational instructions." (Section 4.1 Illustrative supervised tasks)
- "Copy: x_1 x_2 x_3 \dots x_n \longrightarrow x_1 x_2 x_3 \dots x_n" and "Reverse: x_1 x_2 x_3 \dots x_n \longrightarrow x_n x_{n-1} x_{n-2} \dots x_1" and "Double: x_1 x_2 x_3 \dots \longrightarrow x_1 x_2 x_3 \dots x_n x_1 x_2 x_3 \dots x_n" (Figure 7: *Memorization* tasks, Section A.2)
- Inference: "Dynamic" attention and "Constructed" state are inferred from Sections 3.1-3.3. Input and output dynamics remain "Not specified in the paper." because explicit sequence-length bounds for these memorization tasks are not stated.

### Task: Partially observed reinforcement learning (Mini Pacman with viewport)
- "**Mini Pacman with viewport** We follow the formulation of Mini Pacman from [26]. Briefly, the agent navigates a maze to collect food while being chased by ghosts." (Section 4.2 Reinforcement learning)
- "However, we implement this task with a viewport: a  $5 \times 5$  window surrounding the agent that comprises the perceptual input. The task is therefore partially observable, since the agent must navigate the space and take in information through this viewport." (Section 4.2 Reinforcement learning)
- "Thus, the agent must predict the dynamics of the ghosts *in memory*, and plan its navigation accordingly, also based on remembered information about which food has already been picked up." (Section 4.2 Reinforcement learning)
- Inference: Input is treated as 3D (x, y, t) from a spatial viewport stream over time; output is treated as a temporal control/action sequence from the RL navigation setup. "Dynamic" attention and "Constructed" state are inferred from Sections 3.1-3.3.

### Task: Relational planning/control in BoxWorld
- "We study a variant of BoxWorld, which is a pixel-based, highly combinatorial reinforcement learning environment that demands relational reasoning-based planning, initially developed in [46]." (Section A.3 Viewport BoxWorld)
- "The agent is denoted by a dark grey pixel, and has four actions: up, down, left, right." (Section A.3 Viewport BoxWorld)
- "To make this task demand relational reasoning in a memory space, the agent only has perceptual access to a  $5 \times 5$  RGB window, or viewport, appended with an extra frame denoting the color of the key currently in possession." (Section A.3 Viewport BoxWorld)
- "For training we used solution path lengths of at least 1 and up to 5, ensuring that an untrained agent would have a small probability of reaching the goal by chance, at least on the easier levels." (Section A.3 Viewport BoxWorld)
- Inference: Input/output are represented as spatiotemporal observations and action trajectories (3D (x, y, t) input; 1D (t) output). "Capped" output dynamics are inferred from bounded training path lengths and episode termination conditions in Section A.3. "Dynamic" attention and "Constructed" state are inferred from Sections 3.1-3.3.

### Task: Word language modeling
- "Finally, we investigate the task of word-based language modeling. We model the conditional probability  $p(w_t|w_{< t})$  of a word  $w_t$  given a sequence of observed words  $w_{< t} = (w_{t-1}, w_{t-2}, \ldots, w_1)$ ." (Section 4.3 Language Modeling)
- "As a sequential reasoning task, language modeling allows us to assess the RMC's ability to process information over time on a large quantity of natural data, and compare it to well-tuned models." (Section 4.3 Language Modeling)
- Inference: Open dynamics are inferred from recurrent sequence modeling over ongoing text context; this is supported by analysis over varying unroll lengths, including "Perplexities are compared against the 'best' perplexity where the model is unrolled continuously over the full test set." (Figure 12, Section A.4).
