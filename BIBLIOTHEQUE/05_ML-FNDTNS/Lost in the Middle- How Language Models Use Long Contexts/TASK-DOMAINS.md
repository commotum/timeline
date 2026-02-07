# Lost in the Middle: How Language Models Use Long Contexts (Not specified in the paper)
Source: Lost in the Middle- How Language Models Use Long Contexts.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Multi-document question answering | Question and documents (passages) | 1D (t) | Capped | Static | Direct (inferred) | Answer text | 1D (t) | Not specified in the paper. |
| Key-value retrieval | JSON key-value pairs and key | 1D (t) | Capped | Static | Direct (inferred) | Value string | 1D (t) | Not specified in the paper. |
| Open-domain question answering | Query and retrieved documents | 1D (t) | Capped | Static | Direct (inferred) | Answer text | 1D (t) | Not specified in the paper. |

## Summary
The paper studies text-only tasks that require using long contexts: multi-document question answering, synthetic key-value retrieval, and open-domain question answering. Inputs and outputs are token sequences (1D), with inputs bounded by model context windows; outputs are text answers or values. The setups use fixed prompts (static attention), and the model behavior is treated as direct input-to-output mapping (state dynamic inferred from prompting description).

## Evidence
### Task: Multi-document question answering
- "In the multi-document question answering task, the model inputs are (i) a question to answer and (ii) k documents" (§2.1 Experimental Setup)
- "This task requires the model to access the document that contains the answer within its input context and use it to answer the question." (§2.1 Experimental Setup)
- Inference: State Dynamic marked Direct because the paper says "all relevant task specification and data to process is formatted as a textual input context, and the model returns a generated text completion." (§1 Introduction)

### Task: Key-value retrieval
- "In our synthetic key-value retrieval task, the inputs are (i) a string-serialized JSON object with k key-value pairs" (§3.1 Experimental Setup)
- "The goal is to return the value associated with the specified key." (§3.1 Experimental Setup)
- Inference: State Dynamic marked Direct because the paper says "all relevant task specification and data to process is formatted as a textual input context, and the model returns a generated text completion." (§1 Introduction)

### Task: Open-domain question answering
- "takes an input query from NaturalQuestions-Open and returns the k documents from Wikipedia with the highest relevance score." (§5)
- "To condition language models on these retrieved documents, we simply include them in the prompt." (§5)
- "reader accuracy (whether any of the annotated answers appear in the predicted output)" (§5)
- Inference: State Dynamic marked Direct because the paper says "all relevant task specification and data to process is formatted as a textual input context, and the model returns a generated text completion." (§1 Introduction)
