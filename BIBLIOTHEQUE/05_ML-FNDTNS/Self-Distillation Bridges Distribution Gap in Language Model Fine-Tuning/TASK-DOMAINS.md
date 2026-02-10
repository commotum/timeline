# Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning (2024)
Source: Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Response rewriting (self-distillation) | Natural language instruction plus reference answer text | 1D (t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Distilled response text intended to preserve semantics | 1D (t) | Not specified in the paper. |
| Mathematical reasoning | Arithmetic word-problem instructions in natural language | 1D (t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Reasoning response text with final numeric answer | 1D (t) | Not specified in the paper. |
| Tool using (function calling) | Natural language request plus tool/function specification text | 1D (t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Function-call string response | 1D (t) | Not specified in the paper. |
| Code generation | Coding-related prompts/instructions in text | 1D (t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Generated code/text response | 1D (t) | Not specified in the paper. |
| Multi-task instruction following | Diverse natural-language instructions (e.g., question-answering, information extraction, summarization, arithmetic, coding) | 1D (t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Instruction-following text responses | 1D (t) | Not specified in the paper. |
| Safety-aligned response generation | Harmful-behavior instructions, including jailbreak-style prompts with adversarial suffixes | 1D (t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Model response text evaluated for safety | 1D (t) | Not specified in the paper. |
| General-knowledge question answering | Benchmark question prompts from OpenLLM Leaderboard tasks | 1D (t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Answer/response text for factual and commonsense knowledge | 1D (t) | Not specified in the paper. |

## Summary
The paper covers text-based LLM tasks centered on instruction-to-response generation, with specialized downstream tasks in mathematical reasoning, tool use, and code generation, plus evaluations on safety/helpfulness and general knowledge. Across all supported tasks, the input and output are described as natural-language instructions/responses or function-call/code text, which supports a 1D (t) domain classification. The OCR does not explicitly specify interface-level dynamics limits or runtime attention/state behavior, so those fields remain not specified.

## Evidence
### Task: Response rewriting (self-distillation)
- "SDFT first prompts the seed LM to generate responses that uphold semantic equivalence with the original responses present in the task dataset, resulting in the distilled dataset." (Section 1 Introduction)
- "As depicted in Figure 2, the initial step of SDFT involves prompting the seed LM to rewrite the original response  $y^t$  into  $\tilde{y}$" (Section 3.2 Self-Distillation Fine-Tuning)

### Task: Mathematical reasoning
- "These benchmarks encompass: (1) diverse downstream tasks, including mathematical reasoning, tool using and code generation" (Section 1 Introduction)
- "The mathematical reasoning capabilities are improved using the GSM8K dataset (Cobbe et al., 2021), which comprises 8.8k high-quality arithmetic word problems designed at grade school level." (Section 4.2 Datasets for Fine-tuning and Evaluation)

### Task: Tool using (function calling)
- "For single-task datasets, we explore boosting the mathematical reasoning, tool using, and code generation capabilities of LMs during fine-tuning." (Section 4.2 Datasets for Fine-tuning and Evaluation)
- "The tool using proficiency is assessed by leveraging function-calling datasets such as the Gorilla OpenFunctions dataset (Patil et al., 2023)." (Section 4.2 Datasets for Fine-tuning and Evaluation)

### Task: Code generation
- "Following that, we conduct a comparative analysis of the experimental results obtained from vanilla fine-tuning and our proposed SDFT approach across various tasks, encompassing mathematical reasoning, code generation, and tool using." (Section 4 Experiments)
- "Additionally, code generation skills are boosted using the MagiCoder dataset (Wei et al., 2023), while evaluation is conducted using the HumanEval dataset (Chen et al., 2021)." (Section 4.2 Datasets for Fine-tuning and Evaluation)

### Task: Multi-task instruction following
- "The seed LM typically undergoes general SFT, indicating its capacity to map any natural language instruction  $x \in X$  contextualized by the task description  $c \in C$ , to its corresponding output  $g \in Y$ ." (Section 3.1 Fine-tuning LLMs)
- "The Dolly dataset is composed of seven distinct tasks, such as open question & answer, information extraction, and summarization." (Section 4.2 Datasets for Fine-tuning and Evaluation)

### Task: Safety-aligned response generation
- "Safety evaluation. We utilize the harmful behavior instructions from the Advbench dataset (Zou et al., 2023) for evaluation, assessing the safety of models' outputs through keyword matching following Qi et al. (2024)." (Section 4.2 Datasets for Fine-tuning and Evaluation)
- "Additionally, we simulate jailbreaking attempts by appending adversarial suffixes to instructions as illustrated in Zou et al. (2023)." (Section 4.2 Datasets for Fine-tuning and Evaluation)

### Task: General-knowledge question answering
- "Knowledge evaluation. LMs' general knowledge was assessed through evaluations using benchmarks from the OpenLLM Leaderboard, specifically MMLU (Hendrycks et al., 2021), TruthfulQA (Lin et al., 2021), ARC (Clark et al., 2018), HellaSwag (Zellers et al., 2019), and Winogrande (Sakaguchi et al., 2021)." (Section 4.2 Datasets for Fine-tuning and Evaluation)
- "These datasets provide a measure of the models' factual and commonsense knowledge spanning a variety of domains." (Section 4.2 Datasets for Fine-tuning and Evaluation)
