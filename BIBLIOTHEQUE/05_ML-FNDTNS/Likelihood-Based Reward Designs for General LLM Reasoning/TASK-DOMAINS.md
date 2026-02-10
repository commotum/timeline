# Likelihood-Based Reward Designs for General LLM Reasoning (2026)
Source: Likelihood-Based Reward Designs for General LLM Reasoning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Generation (verifiable mathematical question answering) | Question prompts from MATH and DeepScaleR (text tokens) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | Chain-of-thought tokens and short final-answer tokens | 1D (t) (inferred) | Capped |
| Generation (non-verifiable theorem-proof generation) | Theorem/problem prompts from NuminaProof (text tokens) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | Chain-of-thought tokens and long-form proof tokens | 1D (t) (inferred) | Capped |
| Generation (non-verifiable instruction-response generation) | Instruction/question prompts from Alpaca (text tokens) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | Chain-of-thought tokens and long-form response tokens | 1D (t) (inferred) | Capped |

## Summary
The paper covers text generation tasks in both verifiable and non-verifiable reasoning settings: short-answer math QA, long-form theorem-proof generation, and long-form instruction-response generation. Across all tasks, inputs and outputs are token sequences, which maps to 1D (t) (inferred). Output dynamics are Capped due to the explicit maximum generation length (T=1024), while input dynamics are not explicitly specified. Attention is Static (inferred), and state is Constructed (inferred) because the model explicitly generates an intermediate chain-of-thought before the final answer.

## Evidence
### Task: Generation (verifiable mathematical question answering)
- "Datasets. We consider two *verifiable* math benchmarks and two *non-verifiable* long-form datasets." (Section 3.1 Setup: Datasets, Models, and Protocol)
- "(i) MATH (Hendrycks et al., 2021b):We report accuracy on the official test split. The resulting training set contains ~7,000 short-answer problems. (ii) DeepScaleR (Preview) (Luo et al., 2025): we hold out a random 10% for validation to report performance. The training set has ~39,000 short-answer problems." (Section 3.1 Setup: Datasets, Models, and Protocol)
- "For each prompt p, the fine-tuned model should first print a CoT z, then an answer a." (Section 2 Method)
- "At each training step, each process receives a question prompt, and generates G completions with a maximum length of T tokens to that question. Unless noted otherwise, we use G=32 in verifiable domains, and G=4 in nonverifiable domains, and T=1024." (Section B Experimental details)
- Inference: `In Dimension`/`Out Dimension` are `1D (t) (inferred)` because the task is over prompts, CoT traces, and answers as token sequences (Sections 2 and B). `Attention Dynamic` is `Static (inferred)` because no runtime retrieval/observation-selection mechanism is described. `State Dynamic` is `Constructed (inferred)` because the model explicitly constructs an intermediate CoT `z` before generating `a` (Section 2).

### Task: Generation (non-verifiable theorem-proof generation)
- "This paradigm works well in verifiable domains such as mathematics and programming, where ground-truth correctness is available (Cobbe et al., 2021; Hendrycks et al., 2021b; Chen et al., 2021; Austin et al., 2021; Hendrycks et al., 2021a), but it does not naturally extend to non-verifiable domains like long-form proofs or open-ended generation." (Section 1 Introduction)
- "(iv) NuminaProof: starting from NuminaMath-1.5 (Li et al., 2024), we filter for theorem-proof style items. We reserve 1,000 examples for validation, yielding ~50,000 long-form training samples." (Section 3.1 Setup: Datasets, Models, and Protocol)
- "For each prompt p, the fine-tuned model should first print a CoT z, then an answer a." (Section 2 Method)
- "At each training step, each process receives a question prompt, and generates G completions with a maximum length of T tokens to that question. Unless noted otherwise, we use G=32 in verifiable domains, and G=4 in nonverifiable domains, and T=1024." (Section B Experimental details)
- Inference: `In Dimension`/`Out Dimension` are `1D (t) (inferred)` from prompt-to-proof token-sequence generation. `Attention Dynamic` is `Static (inferred)` because no adaptive retrieval/observation control is specified. `State Dynamic` is `Constructed (inferred)` because the CoT is explicitly generated first and used before the final answer (Section 2).

### Task: Generation (non-verifiable instruction-response generation)
- "(iii) Alpaca (cleaned) (Taori et al., 2023): we use the standard cleaned variant; 1,000 random examples are used for validation, leaving ~50,000 training samples with predominantly long-form answers." (Section 3.1 Setup: Datasets, Models, and Protocol)
- "that reference answers are available for situations in which 0/1 rewards are not, such as long-form question-answering." (Section 1 Introduction)
- "For each prompt p, the fine-tuned model should first print a CoT z, then an answer a." (Section 2 Method)
- "At each training step, each process receives a question prompt, and generates G completions with a maximum length of T tokens to that question. Unless noted otherwise, we use G=32 in verifiable domains, and G=4 in nonverifiable domains, and T=1024." (Section B Experimental details)
- Inference: `In Dimension`/`Out Dimension` are `1D (t) (inferred)` since the task operates on text prompts and generated text answers. `Attention Dynamic` is `Static (inferred)` because the paper describes fixed prompt-conditioned generation without dynamic retrieval. `State Dynamic` is `Constructed (inferred)` because the model constructs a CoT before producing the answer (Section 2).
