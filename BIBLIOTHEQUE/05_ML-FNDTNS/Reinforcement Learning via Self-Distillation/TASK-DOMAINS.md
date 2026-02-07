# Reinforcement Learning via Self-Distillation (Not specified in the paper)
Source: Reinforcement Learning via Self-Distillation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Science Q&A (Chemistry, Physics, Biology, Materials science) multiple-choice | Question with four options (scientific Q&A text) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Answer letter (A/B/C/D) | 0D (inferred) | Fixed (inferred) |
| Tool use (tool-call selection) | Tool-API specification + user request (text) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Tool call (function call in XML tags) | 1D (t) (inferred) | Not specified in the paper. |
| Competitive programming / coding problem solving (LiveCodeBench) | Coding problem statement (text) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Python program (code) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper evaluates SDPO on text-based tasks: scientific multiple-choice Q&A, tool-API call selection, and competitive programming/code generation on LiveCodeBench. Inputs are textual prompts (questions, tool specifications, or coding problems), and outputs are either a single answer label or generated tool calls/programs, so the identifiable dimensions are 1D (t) for text and 0D for the MCQ label. The paper does not specify interface dynamics or attention/state behavior for these tasks, so those fields remain unspecified.

## Evidence
### Task: Science Q&A (Chemistry, Physics, Biology, Materials science) multiple-choice
- "Science Q&A (Chemistry, Physics, Biology, Materials science): Undergraduate-level scientific reasoning using reasoning subsets (L3) from SciKnowEval (Feng et al., 2024)." (Section 3.1 Experimental setting)
- "Given a question and four options, please select the right answer." (Appendix E.3 User Templates)
- "For the answer, only output the letter corresponding to the correct option (A, B, C, or D), and nothing else." (Appendix E.3 User Templates)
- Inference: Treated the question/options as a text sequence (1D (t)) and the answer letter as a single label (0D) with fixed size, based on the prompt format. (Appendix E.3 User Templates)

### Task: Tool use (tool-call selection)
- "Tool use: Mapping a tool-API specification and user request to the correct tool call, using ToolAlpaca (Tang et al., 2023)." (Section 3.1 Experimental setting)
- "You are provided with function signatures within <functions></functions> XML tags." (Appendix E.3 User Templates)
- "Output any function calls within < function_calls></function_calls> XML tags." (Appendix E.3 User Templates)
- Inference: Treated the tool specifications, user request, and function-call outputs as text sequences (1D (t)) because they are provided and returned in prompt text. (Appendix E.3 User Templates)

### Task: Competitive programming / coding problem solving (LiveCodeBench)
- "We next evaluate SDPO on coding tasks." (Section 4 Learning with Rich Environment Feedback)
- "LiveCodeBench (LCB; Jain et al., 2025) provides a set of contest-style coding problems, ranging from simple to competition-level." (Section 4 Learning with Rich Environment Feedback)
- "You will be given a coding problem, and you need to write a correct Python program that matches the specification and passes all tests." (Appendix F.2 Examples)
- "In the end, please provide the complete code in a code block enclosed with `````." (Appendix F.2 Examples)
- Inference: Treated the problem statement and program output as text sequences (1D (t)). (Appendix F.2 Examples)