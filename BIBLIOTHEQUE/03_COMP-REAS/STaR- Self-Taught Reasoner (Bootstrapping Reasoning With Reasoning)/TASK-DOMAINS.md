# STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning (2022)
Source: STaR- Self-Taught Reasoner (Bootstrapping Reasoning With Reasoning).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arithmetic addition with rationale generation | Two *n*-digit integer addition problems in prompt format with few-shot examples | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Scratchpad rationale and final numeric sum answer | 1D (t) (inferred) | Capped (inferred) |
| Commonsense multiple-choice question answering with rationale generation | Natural-language question plus five answer choices (CommonsenseQA) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Generated rationale and selected answer choice | 1D (t) (inferred) | Capped (inferred) |
| Grade-school math word-problem solving with rationale generation | Natural-language grade-school word problems (GSM8K) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Multi-step calculation rationale and final answer | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates STaR on three language-based reasoning tasks: arithmetic addition, commonsense multiple-choice QA, and grade-school math word problems. Across all three, the model is prompted with textual sequences and produces textual rationales followed by final answers, supporting 1D (t) input/output characterization. The OCR does not explicitly define Dimension, Dynamics, Attention, or State labels, but the described prompt-and-generation setup supports inferred Capped dynamics, Static attention at inference, and Constructed state via intermediate rationales.

## Evidence
### Task: Arithmetic addition with rationale generation
- "The arithmetic task is to calculate the sum of two *n*-digit integers." (Section 4.2 Datasets)
- "the model is asked to generate the scratchpad (start/end indicated by \"<scratch>\") and the final answer, as in [5]." (Section 4.2 Datasets)
- Inference: `1D (t)` is inferred because the arithmetic examples are provided as text sequences (e.g., "Input: 6 2 4 + 2 5 9"). `Capped` is inferred from sequence-length limits ("batch size of 8 sequences, each of length 1024," Section H). `Static` attention is inferred because the model is prompted with a predefined context per example. `Constructed` state is inferred because the model explicitly generates an intermediate scratchpad/rationale before the final answer.

### Task: Commonsense multiple-choice question answering with rationale generation
- "For commonsense question-answering we follow [13, 6] and use CommonsenseQA (CQA), a widely used multiple-choice dataset for this domain [10]." (Section 4 Experiments)
- "The dataset has 12,247 questions, each with five choices" (Section 4.2 Datasets)
- "prompt the model to generate the rationale and answer for that question." (Section 4.1 Experimental Protocol)
- Inference: `1D (t)` is inferred because CQA inputs and outputs are natural-language question/choice/answer text. `Capped` is inferred from the same 1024-token sequence interface (Section H). `Static` attention is inferred from fixed per-example prompting. `Constructed` state is inferred because rationale text is generated as an intermediate representation before answer emission.

### Task: Grade-school math word-problem solving with rationale generation
- "For grade school math, we use GSM8K" (Section 4 Experiments)
- "These math problems are posed in natural language and require two to eight calculation steps to arrive at a final answer." (Section 4.2 Datasets)
- "We include the following few-shot prompts for GSM8K" (Appendix I GSM8K Few-shot Prompt)
- Inference: `1D (t)` is inferred from natural-language word-problem and textual solution format. `Capped` is inferred from the model's sequence-length-bound interface (Section H). `Static` attention is inferred from fixed prompt-based inference. `Constructed` state is inferred because the model outputs stepwise calculations/rationales before final answers.
