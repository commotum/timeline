# Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities? (2025)
Source: Revisiting the Test-Time Scaling of o1-like Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Generation (math reasoning answer) | Math question text (tokens) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Boxed final answer text (number/expression) | 1D (t) (inferred) | Capped |
| Classification (multiple-choice science QA) | GPQA question text with answer options (tokens) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Single option letter in boxed{} | 0D (inferred) | Fixed |

## Summary
The paper evaluates o1-like language models on text-based reasoning tasks: open-ended mathematical problem answering and multiple-choice scientific QA. The supported inputs are 1D token sequences, with outputs spanning both 1D answer text and 0D label-style choices. The paper explicitly reports a capped generation interface (max length 32k), and the attention/state assignments are inferred as static/direct from the described prompting and scaling setup.

## Evidence
### Task: Generation (math reasoning answer)
- "While MATH-500, AIME, and Omini-MATH focus on mathematical reasoning, GPQA encompasses broader scientific domains." (Section 3, Benchmark)
- "Instruction for MATH-500, AIME and Omini-MATH:" and "Answer the question and enclose the final answer in boxed{}" (Section E Prompt / Instruction)
- "the maximum generation length set to 32k." (Section 3, Models)
- Inference: `1D (t)`, `Capped`, `Static`, and `Direct` are inferred from the text-only prompting/evaluation setup plus bounded generation and context-length scaling: "Sequential scaling increase test-time compute by scaling the length of Chain-of-Thought (CoT)" and "parallel scaling parallely samples multiple solutions and pick the best one." (Section 1 Introduction).

### Task: Classification (multiple-choice science QA)
- "GPQA encompasses broader scientific domains." (Section 3, Benchmark)
- "Instruction for GPQA:" and "Select the best answer from the following options. Output only the letter corresponding to the correct answer, enclosed in boxed{}." (Section E Prompt / Instruction)
- "the maximum generation length set to 32k." (Section 3, Models)
- Inference: `1D (t)` input and `Capped` input dynamics are inferred from the same text-sequence interface used across benchmarks; `0D` output and `Fixed` out dynamics are supported by the requirement to return only one option letter; `Static` attention and `Direct` state are inferred because the described process uses fixed prompt/context processing without explicit runtime retrieval or maintained external memory structures.
