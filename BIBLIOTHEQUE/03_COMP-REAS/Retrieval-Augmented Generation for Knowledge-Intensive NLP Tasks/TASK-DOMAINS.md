# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020)
Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Open-domain question answering | Question text tokens | 1D (t) | Capped (inferred) | Dynamic | Constructed (inferred) | Answer text tokens | 1D (t) | Capped (inferred) |
| Abstractive question answering | Question text tokens | 1D (t) | Capped (inferred) | Dynamic | Constructed (inferred) | Free-form answer text tokens | 1D (t) | Capped (inferred) |
| Question generation (Jeopardy) | Answer-entity text tokens | 1D (t) | Capped (inferred) | Dynamic | Constructed (inferred) | Jeopardy question text tokens | 1D (t) | Capped (inferred) |
| Fact verification classification | Natural-language claim tokens | 1D (t) | Capped (inferred) | Dynamic | Constructed (inferred) | Class label supports/refutes/not enough info | 0D | Fixed |

## Summary
The paper evaluates RAG on four knowledge-intensive NLP tasks: open-domain QA, abstractive QA, Jeopardy question generation, and FEVER fact verification classification. All task interfaces are text-based sequence processing (1D (t)) on input, with sequence generation outputs (1D (t)) for QA and question generation, and label output (0D) for fact verification. The model’s attention is Dynamic across tasks because it retrieves and conditions on top-k documents at runtime, including per-token document use in RAG-Token. Input/output sequence dynamics are marked Capped (inferred) and state as Constructed (inferred) from the retrieval-augmented latent-document architecture and fixed top-k retrieval interface.

## Evidence
### Task: Open-domain question answering
- "Open-domain question answering (QA) is an important real-world application and common testbed for knowledge-intensive tasks [20]." (Section 3.1 Open-domain Question Answering)
- "We treat questions and answers as input-output text pairs (x,y) and train RAG by directly minimizing the negative log-likelihood of answers." (Section 3.1 Open-domain Question Answering)
- Inference: In Dynamics is Capped (inferred), Out Dynamics is Capped (inferred), and State Dynamic is Constructed (inferred) because the model uses "(top-K truncated) distributions over text passages" and treats retrieval as a latent variable while generating from retrieved memory (Sections 2 and 2.1).

### Task: Abstractive question answering
- "RAG models can go beyond simple extractive QA and answer questions with free-form, abstractive text generation." (Section 3.2 Abstractive Question Answering)
- "The task consists of questions, ten gold passages retrieved from a search engine for each question, and a full sentence answer annotated from the retrieved passages." (Section 3.2 Abstractive Question Answering)
- Inference: In Dynamics is Capped (inferred), Out Dynamics is Capped (inferred), and State Dynamic is Constructed (inferred) from the same top-k latent-document retrieval-and-generation mechanism used across tasks (Sections 2 and 2.1).

### Task: Question generation (Jeopardy)
- "To evaluate RAG's generation abilities in a non-QA setting, we study open-domain question generation." (Section 3.3 Jeopardy Ouestion Generation)
- "As Jeopardy questions are precise, factual statements, generating Jeopardy questions conditioned on their answer entities constitutes a challenging knowledge-intensive generation task." (Section 3.3 Jeopardy Ouestion Generation)
- Inference: In Dynamics is Capped (inferred), Out Dynamics is Capped (inferred), and State Dynamic is Constructed (inferred) because generation is performed through the same top-k retrieved latent documents and seq2seq decoding pipeline (Sections 2 and 2.1).

### Task: Fact verification classification
- "FEVER [56] requires classifying whether a natural language claim is supported or refuted by Wikipedia, or whether there is not enough information to decide." (Section 3.4 Fact Verification)
- "We map FEVER class labels (supports, refutes, or not enough info) to single output tokens and directly train with claim-class pairs." (Section 3.4 Fact Verification)
- Inference: In Dynamics is Capped (inferred) and State Dynamic is Constructed (inferred) from the same retrieval-based latent-document architecture; Output Dynamics is Fixed because the task predicts one label from a predefined class set per claim (Sections 3.4 and 2.1).
