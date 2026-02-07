# MEMORIZING TRANSFORMERS (Not specified in the paper)
Source: Memorizing Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language modeling (next-token prediction) | Tokens | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Tokens (next-token prediction) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates language modeling over long-form text corpora spanning books, web articles, math papers, source code, and formal proofs. It describes tokenized input sequences with next-token prediction outputs, implying a 1D (t) token stream with capped context and memory per step. Based on the kNN retrieval mechanism and external memory, attention is dynamic and the model maintains constructed state beyond the raw input.

## Evidence
### Task: Language modeling (next-token prediction)
- "on five language modeling tasks, all of which involve long-form text" (Section 4 Experiments)
- "English language books (PG-19), long web articles (C4), technical math papers (arXiv Math), source code (Github), and formal theorems (Isabelle)." (Section 4 Experiments)
- "The input text is tokenized, and the tokens are embedded into vector space." (Section 3 Method)
- "the token embeddings of the last layer are used to predict the next token." (Section 3 Method)
- Inference: Labeled 1D (t) and Capped because the model processes ordered token subsequences of fixed length and bounded memory: "Long documents are split into subsequences of 512 tokens" and the "external memory keeps a cache of the prior M (key, value) pairs" (Section 3 Method; Section 3.1). Attention marked Dynamic because it performs "approximate k-nearest-neighbor search into the external memory" and retrieved memories "contain a different set of (key, value) pairs for each query" (Section 3.1). State marked Constructed because the model can "memorize the internal representations of past inputs" (Abstract).
