# The Rotary Position Embedding May Cause Dimension Inefficiency in Attention Heads for Long-Distance Retrieval (Not specified in the paper.)
Source: The Rotary Position Embedding May Cause Dimension Inefficiency.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Long-distance key-value retrieval | Query vector `q_i` and randomly sampled key-value pairs `(K, V)` | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Retrieved value `v_i` | 0D (inferred) | Fixed (inferred) |
| Long-context question answering (answer generation) | Instruction, 20 documents, and question tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Generated answer tokens | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper covers two tasks: a synthetic long-distance key-value retrieval setup and a real long-context question-answering setup. Both tasks are text/sequence-oriented and are best mapped to 1D (t) inputs, with static attention over provided context and direct state usage (inferred). The controlled retrieval setup is fixed-size in this paper and produces a single retrieved value, while the QA setup produces token-sequence answers. The QA output-length dynamics are not explicitly specified in the OCR text.

## Evidence
### Task: Long-distance key-value retrieval
- "the attention head can retrieve  $v_i$  with  $q_i$" (Section 4 Controlled Experiment)
- "from any randomly sampled subset of key-value pairs" (Section 4 Controlled Experiment)
- "We sample 128 out of 1000 key-value pairs for the K,V in Eq. 3." (Section A Details of the Controlled Experiment in §4)
- Inference: `1D (t)` input dimension is inferred from indexed/query-key attention over sampled key-value elements and positional setup; `Fixed` input/output dynamics are inferred from the fixed sampling setup (128 pairs); `Static` attention is inferred because the head processes the provided subset rather than selecting an external context; `Direct` state is inferred because the task is reactive retrieval from current `(q_i, K, V)` without an explicit constructed external state; `0D` output dimension is inferred because one target value `v_i` is retrieved per query.

### Task: Long-context question answering (answer generation)
- "we choose a task that involves long dependence modeling, the long-context question-answering task." (Section 5.1 Experimental Setup)
- "we provide the model with 20 documents for each question, among which only one contains the answer." (Section 5.1 Experimental Setup)
- "Then we feed in the LLM the concatenation of the instruction, the documents, the question, and LLMs' generation, optimizing Eq. 4." (Section 5.2 Utilization of Dimensions)
- Inference: `1D (t)` input/output dimensions are inferred from concatenated text prompts and generated text tokens; `Capped` input dynamics are inferred from long-context LLM interface constraints and bounded context usage in the setup; `Static` attention is inferred because the model attends within a provided prompt context; `Direct` state is inferred because behavior is standard autoregressive prompting without an explicit constructed external state.
