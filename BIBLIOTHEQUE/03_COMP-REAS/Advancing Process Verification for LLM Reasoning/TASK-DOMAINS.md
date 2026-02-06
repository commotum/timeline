# Advancing Process Verification for Large Language Models via Tree-Based Preference Learning (Not specified in the paper.)
Source: Advancing Process Verification for LLM Reasoning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reasoning-path verification / ranking (arithmetic reasoning) | Problem statement and reasoning steps (sequence of steps) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Scalar value (reasoning score) | 0D (inferred) | Fixed (inferred) |
| Reasoning-path verification / ranking (commonsense reasoning) | Problem statement and reasoning steps (sequence of steps) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Scalar value (reasoning score) | 0D (inferred) | Fixed (inferred) |

## Summary
Tree-PLV is evaluated as a verifier that ranks reasoning paths for arithmetic and commonsense reasoning benchmarks (GSM8K/MATH and CSQA/StrategyQA). The input is a problem statement with a sequence of reasoning steps, and the verifier outputs a scalar score, implying 1D (t) inputs and 0D fixed outputs (inferred). The paper does not specify input/output dynamics beyond this, nor does it describe attention or state dynamics.

## Evidence
### Task: Reasoning-path verification / ranking (arithmetic reasoning)
- "For arithmetic reasoning, we utilize GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021)." (Section 3.1 Experimental Setup)
- "GSM8K comprises grade school math problems, whereas MATH includes complex competition-level math problems." (Section 3.1 Experimental Setup)
- "x denotes the initial problem statement,  y^+  is the preferred reasoning sequence" (Section 2.3 Step-Level Pairwise Training)
- "Each solution y consists of a sequence of steps" (Section 2.1 Problem Formulations)
- "These solutions are then ranked by the verifier, and the highest-rated one is selected as the most plausible solution." (Section 2.1 Problem Formulations)
- "The verifier is built upon a large language model with an additional randomly initialized linear layer that outputs a scalar value." (Section 2.3 Step-Level Pairwise Training)
- Inference: Interpreted "sequence of steps" as 1D (t) input and "scalar value" as a 0D fixed output. (Based on the quotes above.)

### Task: Reasoning-path verification / ranking (commonsense reasoning)
- "For commonsense reasoning, we employ CSQA (Talmor et al., 2018) and StrategyQA (Geva et al., 2021)." (Section 3.1 Experimental Setup)
- "CSQA challenges the model with multiplechoice questions that often require reasoning based on complex semantics and prior knowledge." (Section 3.1 Experimental Setup)
- "StrategyQA involves true-or-false questions that demand implicit multi-hop reasoning to derive answers." (Section 3.1 Experimental Setup)
- "x denotes the initial problem statement,  y^+  is the preferred reasoning sequence" (Section 2.3 Step-Level Pairwise Training)
- "Each solution y consists of a sequence of steps" (Section 2.1 Problem Formulations)
- "These solutions are then ranked by the verifier, and the highest-rated one is selected as the most plausible solution." (Section 2.1 Problem Formulations)
- "The verifier is built upon a large language model with an additional randomly initialized linear layer that outputs a scalar value." (Section 2.3 Step-Level Pairwise Training)
- Inference: Interpreted "sequence of steps" as 1D (t) input and "scalar value" as a 0D fixed output. (Based on the quotes above.)

---

## CSV Output (required)
