# Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback (2022)
Source: Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dialogue generation (helpful/harmless assistant) | Multi-turn natural-language chat prompts and instructions | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Assistant response tokens | 1D (t) (inferred) | Capped (inferred) |
| Pairwise preference scoring/ranking | A prompt plus a pair of model-generated responses | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | A preference score or better-vs-worse choice | 0D (inferred) | Fixed (inferred) |
| Question answering and story-completion generation | Benchmark question or context text with optional multiple-choice options | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer or completion tokens | 1D (t) (inferred) | Capped (inferred) |
| Summarization generation | Article or document text embedded in a dialogue prompt | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Summary text | 1D (t) (inferred) | Capped (inferred) |
| Code generation (Python function completion) | Python function signatures and docstrings | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Generated Python function body | 1D (t) (inferred) | Capped (inferred) |
| Out-of-distribution detection for harmful/non-helpful inputs | Prompt activation vector extracted from model layers | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Scalar OOD score or in-distribution decision | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers text-centric tasks spanning generation, ranking, and detection: open-ended assistant dialogue, preference scoring, QA/completion benchmarks, summarization, code completion, and OOD-based harmful-input detection. Most tasks operate over token sequences, so their primary data domain is 1D (t), while scoring/detection outputs are scalar 0D values. The paper explicitly reports context limits (1024/2048 tokens) and response token caps (32/128), supporting predominantly capped dynamics with fixed scalar outputs for scoring tasks. Attention and state dynamics are not named explicitly, so the Static and Direct labels are inferred from the described autoregressive and scoring setups.

## Evidence
### Task: Dialogue generation (helpful/harmless assistant)
- "We apply preference modeling and reinforcement learning from human feedback (RLHF) to finetune language models to act as helpful and harmless assistants." (Section Abstract)
- "People can interact with our models in natural language via chat, and ask for help with any text-based task." (Section 2.1 Task Specification and Crowdworkers)
- "train an RL policy to generate a response to each prompt autoregressively" (Section 4.1 Training Setup)
- Inference: 1D (t), Capped, Static, and Direct are inferred because the task is natural-language token generation and the paper specifies capped interface limits ("context size of 1024 tokens" and "limit on the maximum number of tokens per model response"). (Sections A.2 and B.1)

### Task: Pairwise preference scoring/ranking
- "train a PM to assign a higher score to the 'better' item in each comparison." (Section 4.1 Training Setup)
- "each comparison consists of a prompt followed by a pair of model-generated responses, with a PM score evaluated at the end of each response." (Section 4.1 Training Setup)
- "Our preference models are trained on comparison data, with each data point consisting of a *prompt* and a pair of *responses*." (Section A.2 Preference Modeling)
- Inference: 1D (t), Capped, Static, and Direct are inferred from prompt/response token processing with explicit PM context caps ("context size of 1024 tokens" and 2048 for online PM). 0D/Fixed output is inferred from the single PM score per compared response. (Sections A.2 and 4.1)

### Task: Question answering and story-completion generation
- "We evaluate our models on question answering, commonsense, trivia, and story completion using the benchmarks MMLU ..., Lambada ..., Hellaswag ..., OpenBookQA ..., ARC ..., and TriviaQA ..." (Section 4.6.1 NLP Evaluations)
- "## ARC (Multiple choice)" (Section E Details on NLP Evaluations Formatting and Prompts)
- "This eval has 4 choices per question" (Section E Details on NLP Evaluations Formatting and Prompts)
- Inference: The 1D (t), Capped, Static, and Direct labels are inferred because these evaluations are text-prompted LM responses under the same capped token interface described for the models and RLHF setup. (Sections 4.6.1, A.2, and B.1)

### Task: Summarization generation
- "... fully compatible with training for specialized skills such as python coding and summarization." (Section Abstract)
- "Human: Can you write a summary of this article for me?" (Section 5.2 Summarization as a Specialized Skill)
- "Assistant: Sure, here it is:" (Section 5.2 Summarization as a Specialized Skill)
- Inference: 1D (t), Capped, Static, and Direct are inferred because summarization is expressed as dialogue-token generation in the same capped LM/RLHF interface. (Sections 5.2, A.2, and B.1)

### Task: Code generation (Python function completion)
- "We evaluate models on the HumanEval dataset [Chen et al., 2021], which prompts language models with python function signatures and docstrings." (Section 5.3 Natural Language RLHF on Code-Finetuned Models)
- "Models are tasked with correctly filling in the function body given the context" (Section 5.3 Natural Language RLHF on Code-Finetuned Models)
- Inference: 1D (t), Capped, Static, and Direct are inferred because code completion is still sequence generation over tokenized context, run with the same LM/RLHF pipeline constraints. (Sections 5.3 and B.1)

### Task: Out-of-distribution detection for harmful/non-helpful inputs
- "For a prompt i, we extract a vector of activations of dimension  $d_{\rm model}$  from a layer  $\ell$  and call it  $v_i^\ell \in \mathbb{R}^{d_{\rm model}}$ ." (Section 5.4 Applying Out-of-Distribution Detection to Reject Strange or Harmful Requests)
- "The task is to distinguish between an unseen example of harmlessness and helpfulness data without being explicitly shown any harmlessness data at all." (Section 5.4 Applying Out-of-Distribution Detection to Reject Strange or Harmful Requests)
- "we use a scoring function that takes the input and maps it to a scalar value  $\operatorname{score}(x)$ ." (Section 5.4 Applying Out-of-Distribution Detection to Reject Strange or Harmful Requests)
- Inference: Input/Output dimensions are inferred as 0D because the detector consumes a single activation-vector representation and emits a scalar score per prompt; Fixed dynamics are inferred from this single-score interface. Static and Direct are inferred because the detector applies a fixed scoring rule to provided activations without an adaptive retrieval/state-construction loop. (Section 5.4)
