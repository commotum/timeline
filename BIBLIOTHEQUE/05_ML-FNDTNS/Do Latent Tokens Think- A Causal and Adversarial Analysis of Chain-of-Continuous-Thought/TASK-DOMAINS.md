# Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought (Not specified in the paper)
Source: Do Latent Tokens Think- A Causal and Adversarial Analysis of Chain-of-Continuous-Thought.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Instruction-following response generation (safety steering) | Malicious/safe instruction prompts | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Responses to instructions (safety-steered) | 1D (t) (inferred) | Not specified in the paper. |
| Persona-conditioned response generation | Opinion questions with persona instruction (happy/neutral) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Persona-aligned responses | 1D (t) (inferred) | Not specified in the paper. |
| Multiple-choice question answering | Multiple-choice question with options (A/B/C/D) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Answer option (A/B/C/D) | 0D (inferred) | Not specified in the paper. |
| Open-ended question answering | Question with injected context passage | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Answer text | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper evaluates COCONUT on language tasks spanning instruction/response generation (AdvBench), persona-conditioned response generation (PersonalityEdit), multiple-choice QA (MMLU), and open-ended QA (HotpotQA), plus a ProntoQA task whose details are not described. Inputs and outputs are described as questions, instructions, and answers, supporting 1D (t) text-sequence dimensions (inferred), with multiple-choice outputs treated as 0D labels (inferred). The paper does not specify interface dynamics, attention policy, or state construction for these tasks.

## Evidence
### Task: Not specified in the paper.
- "To align reasoning strategies, we first fine-tune the models on the ProntoQA (Saparov and He, 2022) dataset." (Section 4.2 Experiments)
- "For the steering experiments, each model is trained for 6 epochs on ProntoQA." (Appendix C Training Setups)

### Task: Instruction-following response generation (safety steering)
- "Within each split, the number of malicious and safe samples is balanced. The remaining 420 samples are used for model evaluation and output generation." (Appendix D.1 Datasets for Steering Experiments)
- "designed to shift the model's internal embeddings from unsafe to safe, effectively making it produce valid responses to harmful prompts" (Section 4.3 Results)
- Inference: Treated instruction prompts and responses as 1D (t) text sequences based on instruction framing and output generation in the steering setup.

### Task: Persona-conditioned response generation
- "Since the dataset mainly consists of questions asking for the model's opinions on various topics" (Appendix D.1 Datasets for Steering Experiments)
- "appending the instruction \"Please answer with a very happy and cheerful tone\" to construct the \"happy\" and \"neutral\" variants." (Appendix D.1 Datasets for Steering Experiments)
- Inference: Treated questions/instructions and responses as 1D (t) text sequences based on the opinion-question prompts and persona instruction.

### Task: Multiple-choice question answering
- "For multiple-choice experiments (*option manipulation*), we use the MMLU (Hendrycks et al., 2020) dataset." (Section 5.2 Experiments)
- "Only respond in the format: 'Answer: X' where X is one of A, B, C, or D." (Appendix E.2 Swap Experiments)
- Inference: Treated questions/options as 1D (t) text sequences and the selected option as a 0D label based on the multiple-choice format and constrained answer output.

### Task: Open-ended question answering
- "For open-ended question-answering (*context injection*), we use the HotpotQA (Yang et al., 2018) dataset." (Section 5.2 Experiments)
- "For open-ended question-answering tasks, we prepend a passage containing abundant contextual information" (Section 5.1 Method)
- Inference: Treated the question-plus-passage input and answer as 1D (t) text sequences based on the open-ended QA description.
