# Mesa-Extrapolation: A Weave Position Encoding Method for Enhanced Extrapolation in LLMs (Not specified in the paper.)
Source: Mesa-Extrapolation- A Weave Position Encoding Method for Enhanced Extrapolation in LLMs.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Passkey retrieval | input tokens | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | correct password | 1D (t) (inferred) | Not specified in the paper. |
| Language modeling | input tokens | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | next tokens | 1D (t) (inferred) | Not specified in the paper. |
| Summarization | input tokens | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | summary text | 1D (t) (inferred) | Capped (inferred) |
| LongEval lines task | input tokens | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Single-Document QA | input tokens | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Multi-Document QA | input tokens | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Few-shot Learning | input tokens | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Synthesis Tasks | input tokens | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Code Completion | input tokens | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| NIAH (needle-in-a-haystack) retrieval | input tokens | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | specific piece of information (needle) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper evaluates long-context text tasks including passkey retrieval, language modeling, summarization, LongEval lines, LongBench categories (single- and multi-document QA, few-shot learning, synthesis tasks, and code completion), and NIAH retrieval. Where inputs are described via the method, they are token sequences with inferred 1D (t) structure and capped length due to a stated maximum training length; outputs are specified for passkey retrieval, language modeling, summarization, and NIAH, while LongEval and LongBench outputs are not specified. Attention and state dynamics are not explicitly described in the paper.

## Evidence
### Task: Passkey retrieval
- "We assess the accuracy of Mesa-Extrapolation using the generated passkey dataset." (Section 5.1 Evaluation on Passkey Retrieval Tasks)
- "The LLM is required to find the correct password from the sample." (Appendix B.1 Passkey Retrieval Dataset)
- "Input: s[0:T-1] (input tokens with length T)" (Algorithm 1, Section 4.3 Implementation)
- "The vertical black dashed line indicate the position of maximum training length of the model." (Figure 2 caption)
- Inference: Input is treated as a 1D (t) token sequence with capped dynamics based on the explicit token input and maximum training length; the password output is treated as a 1D (t) token sequence.

### Task: Language modeling
- "We further assess the fluency of Mesa-Extrapolation utilizing the perplexity metric." (Section 5.2 Evaluation on Language Modeling)
- "Input: s[0:T-1] (input tokens with length T)" (Algorithm 1, Section 4.3 Implementation)
- "Output: s[T, T + 1, ...]" (Algorithm 1, Section 4.3 Implementation)
- "The vertical black dashed line indicate the position of maximum training length of the model." (Figure 2 caption)
- Inference: Input and output are treated as 1D (t) token sequences with capped input dynamics based on the explicit token input/output and maximum training length.

### Task: Summarization
- "We conduct a summary task using the GovReport dataset and employ ROUGE [31] (ROUGE-1/2/L) as evaluation metrics." (Section 5.3 Evaluation on Summary of Tasks)
- "the generation of summary text within 1000 tokens" (Appendix C.1 BLEU Results on GovReport)
- "Input: s[0:T-1] (input tokens with length T)" (Algorithm 1, Section 4.3 Implementation)
- "The vertical black dashed line indicate the position of maximum training length of the model." (Figure 2 caption)
- Inference: Input is treated as a 1D (t) token sequence with capped dynamics based on the explicit token input and maximum training length; summary outputs are treated as 1D (t) with capped dynamics based on the stated 1000-token limit.

### Task: LongEval lines task
- "We conduct additional testing on LongEval [21] lines task, a recently prominent evaluation task for long texts." (Appendix C.3 Evaluation on LongEval)
- "Input: s[0:T-1] (input tokens with length T)" (Algorithm 1, Section 4.3 Implementation)
- "The vertical black dashed line indicate the position of maximum training length of the model." (Figure 2 caption)
- Inference: Input is treated as a 1D (t) token sequence with capped dynamics based on the explicit token input and maximum training length.

### Task: Single-Document QA
- "We select LongBench [3] dataset and use 5 major categories of tasks, including Single-Document QA, Multi-Document QA, Few-shot Learning, Synthesis Tasks and Code Completion." (Appendix C.4 Evaluation on LongBench)
- "Input: s[0:T-1] (input tokens with length T)" (Algorithm 1, Section 4.3 Implementation)
- "The vertical black dashed line indicate the position of maximum training length of the model." (Figure 2 caption)
- Inference: Input is treated as a 1D (t) token sequence with capped dynamics based on the explicit token input and maximum training length.

### Task: Multi-Document QA
- "We select LongBench [3] dataset and use 5 major categories of tasks, including Single-Document QA, Multi-Document QA, Few-shot Learning, Synthesis Tasks and Code Completion." (Appendix C.4 Evaluation on LongBench)
- "Input: s[0:T-1] (input tokens with length T)" (Algorithm 1, Section 4.3 Implementation)
- "The vertical black dashed line indicate the position of maximum training length of the model." (Figure 2 caption)
- Inference: Input is treated as a 1D (t) token sequence with capped dynamics based on the explicit token input and maximum training length.

### Task: Few-shot Learning
- "We select LongBench [3] dataset and use 5 major categories of tasks, including Single-Document QA, Multi-Document QA, Few-shot Learning, Synthesis Tasks and Code Completion." (Appendix C.4 Evaluation on LongBench)
- "Input: s[0:T-1] (input tokens with length T)" (Algorithm 1, Section 4.3 Implementation)
- "The vertical black dashed line indicate the position of maximum training length of the model." (Figure 2 caption)
- Inference: Input is treated as a 1D (t) token sequence with capped dynamics based on the explicit token input and maximum training length.

### Task: Synthesis Tasks
- "We select LongBench [3] dataset and use 5 major categories of tasks, including Single-Document QA, Multi-Document QA, Few-shot Learning, Synthesis Tasks and Code Completion." (Appendix C.4 Evaluation on LongBench)
- "Input: s[0:T-1] (input tokens with length T)" (Algorithm 1, Section 4.3 Implementation)
- "The vertical black dashed line indicate the position of maximum training length of the model." (Figure 2 caption)
- Inference: Input is treated as a 1D (t) token sequence with capped dynamics based on the explicit token input and maximum training length.

### Task: Code Completion
- "We select LongBench [3] dataset and use 5 major categories of tasks, including Single-Document QA, Multi-Document QA, Few-shot Learning, Synthesis Tasks and Code Completion." (Appendix C.4 Evaluation on LongBench)
- "Input: s[0:T-1] (input tokens with length T)" (Algorithm 1, Section 4.3 Implementation)
- "The vertical black dashed line indicate the position of maximum training length of the model." (Figure 2 caption)
- Inference: Input is treated as a 1D (t) token sequence with capped dynamics based on the explicit token input and maximum training length.

### Task: NIAH (needle-in-a-haystack) retrieval
- "We further conducted experimental validation on the Ruler datasets [17], focusing on the single-keys NIAH task." (Appendix C.9 Evaluation of Phi-3-mini-128k-instruct Model on Ruler Datasets)
- "The needle-in-a-haystack (NIAH) test assesses the ability to retrieve a specific piece of information (the \"needle\") from long distractor texts (the \"haystack\")." (Appendix C.9 Evaluation of Phi-3-mini-128k-instruct Model on Ruler Datasets)
- "Input: s[0:T-1] (input tokens with length T)" (Algorithm 1, Section 4.3 Implementation)
- "The vertical black dashed line indicate the position of maximum training length of the model." (Figure 2 caption)
- Inference: Input is treated as a 1D (t) token sequence with capped dynamics based on the explicit token input and maximum training length; the retrieved needle is treated as a 1D (t) token sequence.

## CSV Output (required)
task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic
Passkey retrieval,input tokens,1D (t) (inferred),Capped (inferred),Not specified in the paper.,Not specified in the paper.,correct password,1D (t) (inferred),Not specified in the paper.
Language modeling,input tokens,1D (t) (inferred),Capped (inferred),Not specified in the paper.,Not specified in the paper.,next tokens,1D (t) (inferred),Not specified in the paper.
Summarization,input tokens,1D (t) (inferred),Capped (inferred),Not specified in the paper.,Not specified in the paper.,summary text,1D (t) (inferred),Capped (inferred)
LongEval lines task,input tokens,1D (t) (inferred),Capped (inferred),Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.
Single-Document QA,input tokens,1D (t) (inferred),Capped (inferred),Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.
Multi-Document QA,input tokens,1D (t) (inferred),Capped (inferred),Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.
Few-shot Learning,input tokens,1D (t) (inferred),Capped (inferred),Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.
Synthesis Tasks,input tokens,1D (t) (inferred),Capped (inferred),Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.
Code Completion,input tokens,1D (t) (inferred),Capped (inferred),Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.
NIAH (needle-in-a-haystack) retrieval,input tokens,1D (t) (inferred),Capped (inferred),Not specified in the paper.,Not specified in the paper.,specific piece of information (needle),1D (t) (inferred),Not specified in the paper.
