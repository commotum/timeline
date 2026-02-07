# Generative Verifiers: Reward Modeling as Next-Token Prediction (Not specified in the paper)
Source: Generative Verifiers- Reward Modeling as Next-Token Prediction.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification (solution correctness verification) - Last Letter Concatenation | word list problem + candidate solution (tokens) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Yes/No token (correctness) | 0D (inferred) | Fixed (inferred) |
| classification (solution correctness verification) - Word Sorting | word list problem + candidate solution (tokens) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Yes/No token (correctness) | 0D (inferred) | Fixed (inferred) |
| classification (solution correctness verification) - Grade school math (GSM8K) | math word problem + candidate solution (tokens) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Yes/No token (correctness) | 0D (inferred) | Fixed (inferred) |
| generation (solution generation) - Last Letter Concatenation | word list problem (tokens) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | concatenated last-letter string (tokens) | 1D (t) (inferred) | Capped (inferred) |
| generation (solution generation) - Word Sorting | word list problem (tokens) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | sorted word list (tokens) | 1D (t) (inferred) | Capped (inferred) |
| generation (solution generation) - Grade school math (GSM8K) | math word problem (tokens) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | solution text (tokens) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper covers three text-based reasoning domains: Last Letter Concatenation, Word Sorting, and GSM8K grade school math. Inputs and outputs are described as word lists or problem/solution text, which supports 1D token sequence interpretations; verification tasks reduce to single Yes/No correctness tokens. The algorithmic word list tasks specify capped list lengths (2-4 words for training and 5-6 for evaluation), while GSM8K length bounds and attention/state dynamics are not specified.

## Evidence
### Task: classification (solution correctness verification) - Last Letter Concatenation
- "Given a list of words, the task is to concatenate the last letters of each word" (Tasks and Data Generation)
- "We train verifiers on lists of lengths 2-4, and evaluate the verifier on the out-of-distribution (OOD) setting of length 6." (Tasks and Data Generation)
- "problem-solution pairs as input and a single 'Yes' or 'No' token as target" (3.1 Direct Verifier)
- Inference: 1D input and capped input length are inferred from the list-of-words task and explicit list lengths; 0D fixed output is inferred from the single Yes/No token. (Tasks and Data Generation; 3.1 Direct Verifier)

### Task: classification (solution correctness verification) - Word Sorting
- "Given a list of words, sort them in alphabetical order." (Tasks and Data Generation)
- "train verifiers on up to 4-words, and evaluate length-generalization performance on 5 word examples." (Tasks and Data Generation)
- "GenRM predicts whether a solution is correct using a single 'Yes' or 'No' token" (3.1 Direct Verifier)
- Inference: 1D input and capped input length are inferred from the list-of-words task and explicit list lengths; 0D fixed output is inferred from the single Yes/No token. (Tasks and Data Generation; 3.1 Direct Verifier)

### Task: classification (solution correctness verification) - Grade school math (GSM8K)
- "GSM8K [Cobbe et al., 2021] is a widely-used dataset to evaluate grade-school math reasoning capabilities of LLMs." (Tasks and Data Generation)
- "generates an output sequence  $\mathbf{y} = (y_1, y_2, \dots, y_T)$  given a input context  $\mathbf{x}$  (e.g., math problem)" (2 Preliminaries)
- "problem-solution pairs as input and a single 'Yes' or 'No' token as target" (3.1 Direct Verifier)
- Inference: 1D input is inferred from the math-problem text setting; 0D fixed output is inferred from the single Yes/No token. (2 Preliminaries; Tasks and Data Generation; 3.1 Direct Verifier)

### Task: generation (solution generation) - Last Letter Concatenation
- "Given a list of words, the task is to concatenate the last letters of each word" (Tasks and Data Generation)
- "We train verifiers on lists of lengths 2-4, and evaluate the verifier on the out-of-distribution (OOD) setting of length 6." (Tasks and Data Generation)
- "integrates reward modelling, which distinguishes between correct and incorrect solutions, with SFT for generating correct solutions." (3.2 Unifying Generation and Verification)
- "For solution generation as well as LLM-as-a-Judge, we use Gemma 2B for algorithmic tasks and Gemini 1.0 Pro [Team et al., 2023] for GSM8K." (Models)
- Inference: 1D input/output and capped input/output lengths are inferred from the list-of-words task definition and the explicit list length ranges. (Tasks and Data Generation)

### Task: generation (solution generation) - Word Sorting
- "Given a list of words, sort them in alphabetical order." (Tasks and Data Generation)
- "train verifiers on up to 4-words, and evaluate length-generalization performance on 5 word examples." (Tasks and Data Generation)
- "integrates reward modelling, which distinguishes between correct and incorrect solutions, with SFT for generating correct solutions." (3.2 Unifying Generation and Verification)
- "For solution generation as well as LLM-as-a-Judge, we use Gemma 2B for algorithmic tasks and Gemini 1.0 Pro [Team et al., 2023] for GSM8K." (Models)
- Inference: 1D input/output and capped input/output lengths are inferred from the list-of-words task definition and the explicit list length ranges. (Tasks and Data Generation)

### Task: generation (solution generation) - Grade school math (GSM8K)
- "GSM8K [Cobbe et al., 2021] is a widely-used dataset to evaluate grade-school math reasoning capabilities of LLMs." (Tasks and Data Generation)
- "generates an output sequence  $\mathbf{y} = (y_1, y_2, \dots, y_T)$  given a input context  $\mathbf{x}$  (e.g., math problem)" (2 Preliminaries)
- "integrates reward modelling, which distinguishes between correct and incorrect solutions, with SFT for generating correct solutions." (3.2 Unifying Generation and Verification)
- "For solution generation as well as LLM-as-a-Judge, we use Gemma 2B for algorithmic tasks and Gemini 1.0 Pro [Team et al., 2023] for GSM8K." (Models)
- Inference: 1D input/output is inferred from the problem/solution text setting and the use of SFT to generate correct solutions. (2 Preliminaries; Tasks and Data Generation; 3.2 Unifying Generation and Verification)
