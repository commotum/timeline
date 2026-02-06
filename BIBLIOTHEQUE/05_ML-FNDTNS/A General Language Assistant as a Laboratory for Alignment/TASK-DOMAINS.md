# A General Language Assistant as a Laboratory for Alignment (Not specified in the paper.)
Source: A General Language Assistant as a Laboratory for Alignment.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Open-ended dialogue response generation | Natural language dialogue text (examples, documents, programming code) | 1D (t) (inferred) | Open (inferred) | Not specified in the paper. | Not specified in the paper. | Natural language responses (text) | 1D (t) (inferred) | Open (inferred) |
| HHH comparison evaluations (helpfulness/honesty/harmlessness) | Queries with paired responses (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Preferred response choice (inferred) | 0D (inferred) | Fixed (inferred) |
| TruthfulQA honesty evaluation (MC1) | TruthfulQA evaluation questions (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Selected response (inferred) | 0D (inferred) | Fixed (inferred) |
| Preference modeling pre-training (StackExchange/Reddit/Wikipedia pairs) | Text pairs from Stack Exchange Q/A, Reddit comments, Wikipedia edits | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Preference score (scalar) | 0D (inferred) | Fixed (inferred) |
| Function synthesis (code generation) | Function prompts/specifications (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Generated function code (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Code Correctness (binary discrimination) | Python function code samples | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Correct vs incorrect label | 0D (inferred) | Fixed (inferred) |
| Lambada completion selection (binary) | Text prompts with candidate completions | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Correct completion label | 0D (inferred) | Fixed (inferred) |
| HellaSwag commonsense inference (ranked multiple choice) | Event description with multiple choices | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Most sensible completion | 0D (inferred) | Fixed (inferred) |
| Learn to Summarize (summary preference ranking) | Article text with paired summaries | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Preferred summary / ranking | 0D (inferred) | Fixed (inferred) |
| Ethics judgments (binary: commonsense morality, deontology, justice, virtue) | Action/statement/trait scenario text | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Binary ethical judgment | 0D (inferred) | Fixed (inferred) |
| Ethics Utilitarianism (ranked pleasantness) | Two similar scenarios | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Rank by pleasantness | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers open-ended text dialogue generation alongside code generation and a mix of preference-ranking and binary discrimination evaluations across summarization, commonsense inference, and ethics. All tasks operate over text/code sequences, with dimensions and size constraints inferred from the fixed token context window; open-ended dialogue is described as arbitrary-length. Outputs range from token sequences for generation to scalar/label choices for ranking and classification, while attention and state dynamics are not specified.

## Evidence
### Task: Open-ended dialogue response generation
- "open-ended natural language dialogue" (Section Open-Ended Dialogue Format and Prompting)
- "general inputs of essentially arbitrary length" (Section Open-Ended Dialogue Format and Prompting)
- "allow similarly general responses" (Section Open-Ended Dialogue Format and Prompting)
- Inference: 1D (t) dimensions inferred from "fixed context window of 8192 tokens" (Section Models); Open dynamics inferred from the arbitrary-length input/response statements above.

### Task: HHH comparison evaluations (helpfulness/honesty/harmlessness)
- "comparison evaluations for each category of helpfulness, honesty, harmlessness" (Section 2.2.1 HHH Evaluations and TruthfulQA)
- "around two-hundred comparisons" (Section 2.2.1 HHH Evaluations and TruthfulQA)
- Inference: Comparison-choice 0D/Fixed output inferred from "comparison evaluations"; 1D (t)/Capped inferred from "fixed context window of 8192 tokens" (Section Models).

### Task: TruthfulQA honesty evaluation (MC1)
- "evaluations on TruthfulQA MC1" (Section 2.2.1 HHH Evaluations and TruthfulQA)
- "adversarial TruthfulQA dataset (MC1)" (Figure 6)
- Inference: Question/response I/O and 0D output inferred from the evaluation framing in "TruthfulQA MC1"; 1D (t)/Capped inferred from "fixed context window of 8192 tokens" (Section Models).

### Task: Preference modeling pre-training (StackExchange/Reddit/Wikipedia pairs)
- "Stack Exchange questionanswer pairs, Reddit comments, and Wikipedia edits" (Section 4.2 Finetuning Results and Scaling Trends)
- "pairwise comparisons, with each pair consisting of a 'better' and 'worse' sample." (Section 4.1 PMP and Datasets)
- "predicts a single scalar 'score' r" (Section 3.1 Preference Modeling)
- Inference: 1D (t)/Capped dynamics inferred from "fixed context window of 8192 tokens" (Section Models); 0D Fixed output inferred from the scalar score description above.

### Task: Function synthesis (code generation)
- "python coding evaluations, the Codex HumanEval" (Section 2.2 Evaluations and Alignment Taxes)
- "QuixBugs challenge reformulated as a function synthesis task" (Section 2.2 Evaluations and Alignment Taxes)
- Inference: Code/text sequence I/O inferred from the "function synthesis task" wording; 1D (t)/Capped inferred from "fixed context window of 8192 tokens" (Section Models).

### Task: Code Correctness (binary discrimination)
- "Code Correctness is a dataset we constructed from python functions" (Section 3 Scaling of Preference Modeling vs Imitation Learning)
- "correctness determined by unit tests" (Section 3 Scaling of Preference Modeling vs Imitation Learning)
- Inference: 0D Fixed output inferred from "data has only two possible labels" (Section Scaling of Preference Modeling vs Imitation Learning); 1D (t)/Capped from "fixed context window of 8192 tokens" (Section Models).

### Task: Lambada completion selection (binary)
- "correct answers in the training set" (Section Lambada (Binary))
- "trained the discriminator to identify the correct completion" (Section Lambada (Binary))
- Inference: 0D Fixed output inferred from "data has only two possible labels" (Section Scaling of Preference Modeling vs Imitation Learning); 1D (t)/Capped from "fixed context window of 8192 tokens" (Section Models).

### Task: HellaSwag commonsense inference (ranked multiple choice)
- "multiple choice evaluation on commonsense inference" (Section HellaSwag (Ranked))
- "identify the most sensible completion" (Section HellaSwag (Ranked))
- Inference: 0D Fixed output inferred from the multiple-choice formulation above; 1D (t)/Capped from "fixed context window of 8192 tokens" (Section Models).

### Task: Learn to Summarize (summary preference ranking)
- "collection of articles" (Section Learn to Summarize (Ranked))
- "pair of summaries that have been ranked" (Section Learn to Summarize (Ranked))
- Inference: 0D Fixed output inferred from the ranked-summary formulation above; 1D (t)/Capped from "fixed context window of 8192 tokens" (Section Models).

### Task: Ethics judgments (binary: commonsense morality, deontology, justice, virtue)
- "Assess whether a given action is morally acceptable." (Section Ethics (Binary, except for Utilitarianism))
- "according to a set of rules or constraints." (Section Ethics (Binary, except for Utilitarianism))
- "on the basis of impartiality and desert." (Section Ethics (Binary, except for Utilitarianism))
- "Given a personal trait and a scenario" (Section Ethics (Binary, except for Utilitarianism))
- Inference: 0D Fixed output inferred from "binary classification problems" (Section Scaling of Preference Modeling vs Imitation Learning); 1D (t)/Capped from "fixed context window of 8192 tokens" (Section Models).

### Task: Ethics Utilitarianism (ranked pleasantness)
- "Utilitarianism (ranked)" (Section Ethics (Binary, except for Utilitarianism))
- "Given two similar scenarios, rank them" (Section Ethics (Binary, except for Utilitarianism))
- Inference: 0D Fixed output inferred from the ranking description above; 1D (t)/Capped from "fixed context window of 8192 tokens" (Section Models).
