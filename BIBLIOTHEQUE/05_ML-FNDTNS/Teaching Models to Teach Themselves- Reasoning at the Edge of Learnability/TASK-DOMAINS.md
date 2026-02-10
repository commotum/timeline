# Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability (2026)
Source: Teaching Models to Teach Themselves- Reasoning at the Edge of Learnability.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mathematical reasoning question answering | Math question text tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Reasoning-trace tokens plus boxed final answer tokens | 1D (t) (inferred) | Capped (inferred) |
| Synthetic curriculum generation (question-answer pair generation) | Fixed teacher prompt tokens | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Synthetic math question-answer pair tokens | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper covers two text-domain tasks: solving hard math reasoning problems and generating synthetic math question-answer pairs as a curriculum. Both tasks operate over token sequences, so the justified dimension is 1D (t), with capped sequence interfaces for generation and training. The teacher-generation input is fixed by design (same prompt each outer-loop step), while student problem-solving inputs vary per benchmark instance. Attention and state behavior are not explicitly labeled in the paper and are inferred as static/direct from the fixed-prompt autoregressive setup without runtime retrieval or external memory.

## Evidence
### Task: Mathematical reasoning question answering
- "we focus on math reasoning tasks, where this setting is common. We use three such benchmarks: MATH (Hendrycks et al., 2021), HARP (Yue et al., 2024), and OlympiadBench (He et al., 2024)." (Section 4.1 Models and Datasets)
- "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first shows the complete reasoning process step by step, then provides the final answer in \boxed{}." (Section B.1 Student Prompt)
- Inference: `1D (t)`, `Capped`, `Static`, and `Direct` are inferred from the token-sequence prompt/response formulation and bounded generation setup ("For each problem ... a token budget of 1024 tokens" in Section B.5; "Max generated tokens ... Student 1024" in Table 3).

### Task: Synthetic curriculum generation (question-answer pair generation)
- "The teacher's role is to generate synthetic problems that provide the student with the necessary gradient signal to escape the performance plateau." (Section 3.1 Overview)
- "Since we cannot automatically verify the answers to proposed problems, we prompt the teacher to generate both the question and answer." (Section 3.2 Outer Loop: Teacher Training)
- "Teacher Prompt. At every outer-loop step, the teacher is given the same prompt." (Section B.1 Prompts)
- Inference: `In Dynamics = Fixed` is inferred from the constant teacher prompt each step; `Out Dynamics = Capped` is inferred from bounded generation ("Max generated tokens ... Teacher 512" in Table 3); `1D (t)`, `Static`, and `Direct` are inferred from autoregressive token generation without a described runtime retrieval/observation-selection or persistent external memory mechanism.
