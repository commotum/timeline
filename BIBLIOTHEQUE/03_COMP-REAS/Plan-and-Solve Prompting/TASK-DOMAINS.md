# Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models (Not specified in the paper.)
Source: Plan-and-Solve Prompting.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arithmetic reasoning (math word problems) | Math word problems (text questions) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Numeric answer or option label | 0D (inferred) | Not specified in the paper. |
| Commonsense question answering | Commonsense questions (text) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Option label or Yes/No answer | 0D (inferred) | Not specified in the paper. |
| Symbolic reasoning (last-letter concatenation) | Name-based questions for last-letter concatenation (text) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Concatenated last-letter string | 1D (t) (inferred) | Not specified in the paper. |
| Symbolic reasoning (coin-flip state tracking) | Coin flip questions/steps (text) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Yes/No answer | 0D (inferred) | Not specified in the paper. |

## Summary
The paper evaluates Plan-and-Solve prompting on three text-based reasoning domains: arithmetic word problems, commonsense question answering, and symbolic reasoning (last-letter concatenation and coin-flip tracking). Inputs are natural-language questions, while outputs are single answers (numbers/options/yes-no) except for last-letter concatenation, which yields a short string. Dimensionality is inferred from these text and answer types, while dynamics, attention, and state characteristics are not explicitly specified.

## Evidence
### Task: Arithmetic reasoning (math word problems)
- "Arithmetic Reasoning:** (1) the GSM8K (Cobbe et al., 2021) dataset of high quality linguistically diverse grade school math word problems" (Section 3.1 Benchmarks)
- "| GSM8K        | Math   | 1319      | 46.9       | Number   |" (Table 1)
- "| AQUA         | Math   | 254       | 51.9       | Option   |" (Table 1)
- Inference: Inputs are text word problems, so In Dimension is 1D (t); Table 1 answer types (Number/Option) imply a 0D output. (Section 3.1 Benchmarks; Table 1)

### Task: Commonsense question answering
- "Commonsense Reasoning: (7) the CSQA (Talmor et al., 2019) benchmark dataset of multiple-choice questions" (Section 3.1 Benchmarks)
- "| CSQA         | CS     | 1221      | 27.8       | Option   |" (Table 1)
- "| StrategyQA   | CS     | 2290      | 9.6        | Yes / No |" (Table 1)
- Inference: Inputs are text questions, so In Dimension is 1D (t); Table 1 answer types (Option/Yes-No) imply a 0D output. (Section 3.1 Benchmarks; Table 1)

### Task: Symbolic reasoning (last-letter concatenation)
- "Last Letter Concatenation (Wei et al., 2022b) dataset of questions requiring the last letters of words in a name to be concatenated" (Section 3.1 Benchmarks)
- "| Last Letters | Sym.   | 500       | 15.0       | String   |" (Table 1)
- Inference: Inputs are text questions, so In Dimension is 1D (t); the "String" answer type implies a 1D (t) output. (Section 3.1 Benchmarks; Table 1)

### Task: Symbolic reasoning (coin-flip state tracking)
- "Coin Flip (Wei et al., 2022b) dataset of questions on whether a coin is still heads up" (Section 3.1 Benchmarks)
- "| Coin Flip    | Sym.   | 500       | 37.0       | Yes / No |" (Table 1)
- Inference: Inputs are text questions, so In Dimension is 1D (t); the Yes/No answer type implies a 0D output. (Section 3.1 Benchmarks; Table 1)
