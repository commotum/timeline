# Constitutional AI: Harmlessness from AI Feedback (Not specified in the paper.)
Source: Constitutional AI- Harmlessness from AI Feedback.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generation (assistant responses) | user requests (natural language) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | assistant responses (natural language) | 1D (t) (inferred) | Not specified in the paper. |
| generation (self-critique and revision) | assistant responses + constitutional principles (natural language) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | critiques and revised responses (natural language) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The OCR text describes general-purpose language models/AI assistants that generate natural language responses to user requests, and a constitutional training approach where the model evaluates its own outputs through self-critique and revision. Inputs and outputs are natural language text, so the task domains are text-based sequences (1D (t) inferred), while interface size dynamics, attention dynamics, and state dynamics are not specified. The covered tasks are limited to response generation and self-critique/revision; no other modalities or task types are described.

## Evidence
### Task: generation (assistant responses)
- "CAI enables AI systems to generate useful responses while also minimizing harm." (Section: Constitutional AI: Harmlessness from AI Feedback)
- "These models engage with user requests, but are less likely to help users with unsafe or unethical requests." (Section: ANTHROP\C)
- Inference: In/Out Dimension are 1D (t) because the inputs/outputs are natural language user requests and responses, which are sequential text. (Supported by the quotes above.)

### Task: generation (self-critique and revision)
- "The approach is called Constitutional AI (CAI) because it gives an AI system a set of principles (i.e., a \"constitution\") against which it can evaluate its own outputs." (Section: Constitutional AI: Harmlessness from AI Feedback)
- "The model's self-critique and -revision approach can be framed as reinforcement learning from *AI* feedback (RLAIF)." (Section: ANTHROP\C)
- "Which of these assistant responses is less harmful? Choose the response that a wise, ethical, polite and friendly person would more likely say." (Section: ANTHROP\C)
- Inference: In/Out Dimension are 1D (t) because the principles, responses, critiques, and revisions are described in natural language text, which is sequential. (Supported by the quotes above.)
