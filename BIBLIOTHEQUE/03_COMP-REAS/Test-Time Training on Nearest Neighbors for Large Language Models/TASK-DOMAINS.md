# TEST-TIME TRAINING ON NEAREST NEIGHBORS FOR LARGE LANGUAGE MODELS (Not specified in the paper)
Source: Test-Time Training on Nearest Neighbors for Large Language Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nearest-neighbor retrieval for adaptation | text token sequences (test-instance queries) | 1D (t) | Capped | Dynamic | Constructed | retrieved neighbor text token sequences | 1D (t) | Capped |
| Causal language modeling / next-token prediction | text token sequences (test instance plus retrieved neighbor text for test-time fine-tuning) | 1D (t) | Capped | Dynamic | Constructed | next-token predictions / generated token sequences | 1D (t) | Capped (inferred) |

## Summary
The paper covers two operational tasks: nearest-neighbor retrieval over a large text index and causal language modeling with test-time fine-tuning on retrieved text. Both tasks operate over token sequences, so the supported domain is 1D (t) in both input and output. Dynamics are Capped by explicit maximum sequence lengths and finite neighbor counts per query, while runtime retrieval makes the attention policy Dynamic. Because the system builds and uses an external index and temporarily updates model parameters per test instance, State is Constructed.

## Evidence
### Task: Nearest-neighbor retrieval for adaptation
- "For each test instance, we retrieve its nearest neighbors from a huge database, and fine-tune the model on those neighbors before applying it to the test instance." (Section 1 Introduction)
- "Our distributed index can serve each nearest neighbor query to approximately 200 million vectors and 1TB of data in approximately one second on standard hardware." (Section 1.1 OUR CONTRIBUTIONS)
- "For simplicity, we naively truncate long sequences to the maximum sequence length of the embedding model." (Section 3 NEAREST NEIGHBOR INDEX)
- "Surprisingly, retrieving and training on as few as 20 neighbors, each for only one gradient iteration, drastically improves performance across more than 20 language modeling tasks in the Pile." (Abstract)

### Task: Causal language modeling / next-token prediction
- "We investigate a simple, yet powerful, heuristic in this space, called test-time training on nearest neighbors (TTT-NN), for the task of language modeling." (Section 1 Introduction)
- "We evaluate our method on all 22 tasks for language modeling from the Pile benchmark." (Section 1.1 OUR CONTRIBUTIONS)
- "The basic idea is that autoregressive language modeling maps each context window of tokens to a predicted distribution over the next token." (Section 2.2 LANGUAGE MODELS USING RETRIEVAL)
- "The maximum sequence length of the model is 1048 tokens." (Section A RESULTS FOR GPT-2-SMALL ON SPLIT SEQUENCES)
- Inference: Out Dynamics is labeled "Capped (inferred)" because the paper explicitly states maximum sequence lengths for evaluated models and describes chunking long sequences during processing, indicating bounded per-pass prediction length. (Section 4 TEST-TIME TRAINING ON NEAREST NEIGHBORS; Sections A-C)
