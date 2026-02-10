# Text and Code Embeddings by Contrastive Pre-Training (Not specified in the paper)
Source: Text and Code Embeddings by Contrastive Pre-Training.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Text classification | text tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | class label | 0D (inferred) | Fixed (inferred) |
| Sentence similarity scoring | sentence tokens (pair) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | similarity score | 0D (inferred) | Fixed (inferred) |
| Semantic text search | query tokens; document/passage tokens | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | ranked documents/passages | 1D (t) (inferred) | Capped (inferred) |
| Semantic code search | natural language query tokens; code tokens | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | ranked code blocks | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates one shared embedding framework across four downstream task intents: text classification, sentence similarity scoring, semantic text search, and semantic code search. Inputs are token sequences (text and code), while outputs are either point decisions/scores (classification and similarity) or ranked retrieval results (text/code search). The supported dimensional range is 1D (t) inputs with 0D and 1D (t) outputs, and output dynamics vary from Fixed to Capped depending on the task. Attention is static for direct scoring/classification and dynamic for retrieval tasks, with state treated as constructed through learned embeddings and vector indexing.

## Evidence
### Task: Text classification
- "When evaluated on linear-probe classification, the embeddings are used as features to train a linear classifier to solve a variety of downstream tasks." (Section 3.1.1)
- "In this section, we discuss results using zero-shot classification and k-nearest neighbor classification on the SST-2 binary sentiment classification task (Socher et al., 2013)." (Section 3.1.2)
- "In the first zero-shot experiment, each input text is assigned with one of the two labels ('positive', 'negative') based on which label has its embedding closest to the input text embedding." (Section 3.1.2)
- Inference: Input dimension (1D (t)) and capped input dynamics are inferred from sequence processing: "We insert two special token delimiters, <code>[SOS]</code> and <code>[EOS]</code>, to the start and end of the input sequence respectively." (Section 2.1). Output dimension (0D) and fixed output dynamics are inferred because one label is assigned per input text (Section 3.1.2). Attention is inferred as static because the classifier consumes the given encoded input; state is inferred as constructed because embeddings are explicitly used as features (Sections 2.1, 3.1.1).

### Task: Sentence similarity scoring
- "The SentEval benchmark (Conneau & Kiela, 2018) is widely adopted to assess the quality of sentence embeddings, consisting of a broad collection of tasks in the categories of linear-probe classification and sentence similarity, and we use the same to evaluate ours." (Section 3.1)
- "On sentence similarity tasks in SentEval, we find that our models perform worse than previous SOTA methods (Table 4)." (Section 3.1.3)
- "The Transformer encoder maps the input, x and y, to embeddings,  $v_x$  and  $v_y$  respectively and the similarity between two inputs is quantified by the cosine similarity between their embeddings,  $v_x$  and  $v_y$  (Figure 3)." (Section 2.1)
- Inference: Input dimension (1D (t)) and capped input dynamics are inferred from sequence-based encoder inputs (Section 2.1). Output is inferred as a 0D fixed score because similarity is quantified as one cosine value per input pair (Section 2.1). Attention is inferred static (no runtime retrieval/selection mechanism described for this task), and state is inferred constructed via learned embeddings.

### Task: Semantic text search
- "First, we evaluate our models on several large-scale text search benchmarks. MSMARCO (Nguyen et al., 2016) requires the model to search over 4M documents while Natural Questions (NQ) (Kwiatkowski et al., 2019) and TriviaQA (Joshi et al., 2017) involve searching over 21M Wikipedia documents." (Section 3.2.1)
- "We use the FAISS library (Johnson et al., 2019) to build the vector indices for approximate k-nearest neighbor search." (Section 3.2.1)
- "Table 5. Evaluation of unsupervised cpt-text models of different sizes on several large-scale text search benchmarks. We report MRR@10 on MSMARCO and Recall@20, Recall@100 for NQ and TriviaQA as done in prior work." (Table 5)
- Inference: Input dimension is inferred as 1D (t) from tokenized query/document text; input dynamics are inferred Open because the system is framed as searching over very large, scalable corpora (4M, 21M, and in introduction "millions or billions of items"). Attention is inferred Dynamic because nearest-neighbor retrieval selects which documents are considered at runtime (Sections 3.2.1, 3.2.2). Output is inferred as a capped 1D ranked list due top-K style retrieval metrics (MRR@10, Recall@20/100), and state is inferred Constructed via dense embeddings and offline vector indices.

### Task: Semantic code search
- "We evaluate our code embedding models on the code search task using the CodeSearchNet benchmark (Husain et al., 2020)." (Section 3.3)
- "Given a natural language query, the model is expected to retrieve the relevant code block among 1K candidates." (Section 3.3)
- "We also evaluate on a harder setting of finding the relevant code block among 10K candidates instead of 1K." (Section 3.3)
- Inference: Input dimension (1D (t)) is inferred from language/code token sequences, supported by paired text-code training ("Code embedding models treat the top-level docstring in a function along with its implementation as a (text, code) pair." Section 1). Input and output dynamics are inferred Capped from explicit 1K/10K candidate sets (Section 3.3). Attention is inferred Dynamic because the system retrieves relevant code from candidate pools, and state is inferred Constructed through learned embeddings used for retrieval.
