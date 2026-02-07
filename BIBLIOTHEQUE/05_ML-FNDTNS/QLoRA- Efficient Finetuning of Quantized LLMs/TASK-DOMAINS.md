# QLoRA: Efficient Finetuning of Quantized LLMs (Not specified in the paper.)
Source: QLoRA- Efficient Finetuning of Quantized LLMs.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generation (instruction-following / chatbot responses) | instructions/prompts (including multi-turn dialog history) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | response text | 1D (t) (inferred) | Not specified in the paper. |
| classification (language understanding; MMLU/GLUE) (inferred) | multiple-choice questions or NLU prompts (text) (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | answer choice/label (inferred) | 0D (inferred) | Not specified in the paper. |
| prediction (language modeling) (inferred) | text sequences (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | next-token predictions (inferred) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper focuses on text-based generation tasks via instruction-following and chatbot response generation, and evaluates language-understanding classification on MMLU/GLUE alongside language modeling. Inputs and outputs are described as prompts, responses, and multiple-choice questions, implying 1D sequential text and label outputs for classification (inferred). The paper does not specify attention or state dynamics, nor explicit interface dynamics bounds for these tasks.

## Evidence
### Task: generation (instruction-following / chatbot responses)
- "instruction finetuning uses input-output pairs of various data sources to finetune a pretrained LLM to generate the output given the input as a prompt." (Section Instruction Finetuning)
- "The OASST1 dataset is a multilingual collection of crowd-sourced multiturn dialogs between a user and an assistant." (Section Benchmark Data)
- "We select all user messages in the validation dataset as queries and include previous turns in the prompt." (Section Benchmark Data)
- "The tournament is made up of matches where pairs of models compete to produce the best response for a given prompt." (Section Elo Rating)
- Inference: Treated prompts/responses and dialog turns as 1D (t) sequences of text; dimensionality is not explicitly stated. (Section Instruction Finetuning; Section Benchmark Data)

### Task: classification (language understanding; MMLU/GLUE) (inferred)
- "Our evaluations include GLUE [58] with RoBERTa-large [38], Super-NaturalInstructions (TKInstruct) [61] with T5 [49], and 5-shot MMLU [24] after finetuning LLaMA on Flan v2 [39] and Alpaca [55]." (Section Experimental setup)
- "This is a multiple-choice benchmark covering 57 tasks including elementary mathematics, US history, computer science, law, and more." (Section 5.2 Evaluation)
- "We report 5-shot test accuracy." (Section 5.2 Evaluation)
- Inference: Classified this as a language-understanding classification task with text prompts and label outputs because MMLU is described as multiple-choice and GLUE is reported with accuracy; input/output formats and dimensions are not explicitly specified. (Section Experimental setup; Section 5.2 Evaluation)

### Task: prediction (language modeling) (inferred)
- "quantized LLMs (OPT [72], BLOOM [52], Pythia [7], LLaMA) of different sizes (125M to 65B) with different data types are evaluated on language modeling and a set of zero-shot tasks." (Section 4-bit NormalFloat yields better performance than 4-bit Floating Point)
- Inference: Treated language modeling as next-token prediction over 1D (t) text sequences; the paper does not specify the input/output structure or dynamics. (Section 4-bit NormalFloat yields better performance than 4-bit Floating Point)
