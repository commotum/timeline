# Precise Zero-Shot Dense Retrieval without Relevance Labels (Not specified in the paper.)
Source: Precise Zero-Shot Dense Retrieval without Relevance Labels.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Text generation (hypothetical document) | query text + instruction text | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | hypothetical document/passage text | 1D (t) (inferred) | Not specified in the paper. |
| Document retrieval (dense retrieval / similarity search) | query text; document corpus (text) | 1D (t) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | retrieved documents (text) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper frames HyDE as dense retrieval decomposed into a generative passage-writing step and a document-document similarity search over a corpus. Both tasks operate over textual queries/instructions and documents, so the addressed objects are 1D sequences (inferred). The paper does not specify explicit size constraints for inputs or outputs. The generation step is a direct mapping from fixed inputs, whereas retrieval dynamically selects documents using constructed embedding representations (inferred).

## Evidence
### Task: Text generation (hypothetical document)
- "First, we feed the query to the generative model and instruct it to \"write a document that answers the question\", i.e. a hypothetical document." (Section 1 Introduction)
- "Please write a passage to answer the question" (Appendix A.1.1 Web Search)
- Inference: Marked input/output dimensions as 1D (t) because the task operates on textual queries/instructions and produces textual passages; marked attention as Static and state as Direct because the generation step is described as taking a fixed query+instruction and producing a document without any retrieval or constructed external state. (Section 1 Introduction; Appendix A.1.1 Web Search)

### Task: Document retrieval (dense retrieval / similarity search)
- "Dense retrieval models similarity between query and document with inner product similarity. Given a query q and document d, it uses two encoder function  $\operatorname{enc}_q$  and  $\operatorname{enc}_d$  to map them into d dimension vectors  $\mathbf{v_q}$ ,  $\mathbf{v_d}$ , whose inner product is used as similarity measurement." (Section 3.1 Preliminaries)
- "We use this vector to search against the corpus embeddings. The most similar real documents are retrieved and returned." (Section 1 Introduction)
- Inference: Marked input/output dimensions as 1D (t) because queries and documents are textual; marked attention as Dynamic because the system searches a corpus to select similar documents at runtime; marked state as Constructed because the method encodes documents into embedding vectors and searches corpus embeddings. (Section 3.1 Preliminaries; Section 1 Introduction)
