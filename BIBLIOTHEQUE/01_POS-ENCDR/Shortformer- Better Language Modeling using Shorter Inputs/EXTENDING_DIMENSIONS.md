## 1. Basic Metadata
- Title: "Shortformer: Better Language Modeling Using Shorter Inputs" (Title)
- Authors: "Ofir Press<sup>1,2</sup> Noah A. Smith<sup>1,3</sup> Mike Lewis<sup>2</sup>" (Title/author block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary
The paper claims that "shorter inputs are not harmful" and introduces "two new methods that decrease input length" to improve transformer language modeling efficiency and perplexity (Abstract).

## 3. Tasks Evaluated
Task 1: Language modeling (next-token prediction/perplexity evaluation and token-by-token generation) on WikiText-103.
- Task type: Generation; Other (perplexity evaluation).
- Dataset(s) used: WikiText-103.
- Domain: natural language text (English Wikipedia).
- Evidence: "Our baseline is the Baevski and Auli (2018) model, henceforth B&A, trained and evaluated on WikiText-103 (Merity et al., 2016)." (Section 2, Experimental Setup); "The training set contains 103.2 million tokens from English Wikipedia." (Section 2, Experimental Setup); "In evaluation, a model assigns a perplexity score to a given sequence." (Section 2, Evaluation vs. Generation); "In generation, a model generates a new sequence, as in demonstrations of GPT-3 (Brown et al., 2020)." (Section 2, Evaluation vs. Generation); "Generation is done only with a sliding window with stride S = 1, which we refer to as token-by-token generation." (Section 2, Evaluation vs. Generation)

Task 2: Language modeling (next-token prediction/perplexity evaluation) on Toronto Book Corpus (TBC).
- Task type: Generation; Other (perplexity evaluation).
- Dataset(s) used: Toronto Book Corpus (TBC).
- Domain: natural language text (books).
- Evidence: "To verify that our results transfer to other datasets, we ran our models on the Toronto Book Corpus (TBC) (Zhu et al., 2015), a 700M token collection of books that has previously been used in the training corpus of BERT (along with English Wikipedia)." (A.5 Toronto Book Corpus); "In evaluation, a model assigns a perplexity score to a given sequence." (Section 2, Evaluation vs. Generation)

## 4. Domain and Modality Scope
- Evaluation performed on multiple domains within the same modality (text): "The training set contains 103.2 million tokens from English Wikipedia." (Section 2, Experimental Setup); "a 700M token collection of books" (A.5 Toronto Book Corpus).
- Multiple modalities: Not specified.
- Domain generalization or cross-domain transfer: "To verify that our results transfer to other datasets, we ran our models on the Toronto Book Corpus (TBC) (Zhu et al., 2015), a 700M token collection of books that has previously been used in the training corpus of BERT (along with English Wikipedia)." (A.5 Toronto Book Corpus)

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Language modeling on WikiText-103 | No (trained per dataset; no joint training stated) | Not specified | Not specified | "trained and evaluated on WikiText-103 (Merity et al., 2016)." (Section 2, Experimental Setup) |
| Language modeling on Toronto Book Corpus | No (trained per dataset; no joint training stated) | Not specified | Not specified | "we ran our models on the Toronto Book Corpus (TBC) (Zhu et al., 2015)" (A.5 Toronto Book Corpus) |

## 6. Input and Representation Constraints
- Fixed or variable input resolution: Not specified.
- Fixed patch size: Not specified.
- Fixed number of tokens per input subsequence: "We refer to the list of tokens as the *current input subsequence* (whose length is L)." (Section 2, Background and Experimental Setup); "This model has a subsequence length of 3,072 tokens." (Section 2, Experimental Setup)
- Fixed number of tokens per batch: "of tokens in each batch to 9,216 but vary the subsequence length L and batch size (so the product of the batch size and subsequence length remains at 9,216)." (Section 3.1)
- Cache length constraint: "In all our models with PIA and caching, we set L'=L" (Section 5.3)
- Fixed dimensionality (e.g., strictly 2D): Not specified.
- Padding or resizing requirements: Not specified.

## 7. Context Window and Attention Structure
- Maximum sequence length: "This model has a subsequence length of 3,072 tokens." (Section 2, Experimental Setup); "Subsequence length L is an upper bound on the effective context window at each timestep." (Section 3)
- Fixed or variable sequence length: "but vary the subsequence length L and batch size (so the product of the batch size and subsequence length remains at 9,216)." (Section 3.1).
- Attention type: Global self-attention within subsequences: "self-attention(key=**X**, query=**X**, value=**X**)" (Section 5); caching extends attention to previous subsequence: "we can store and attend to representations of the previous subsequence" (Section 5.2).
- Mechanisms to manage computational cost: "Here, we choose a stride S between 1 and L-1 and advance the window by S tokens after each forward pass." (Section 2, Sliding Window Inference); "Using PIA and caching, we can reuse L-1 of the previous outputs at every layer. Thus, our attention sublayer takes O(L) time (because now there is a single query and L keys)." (Section 5.2); "First, we show that initially training a model on short subsequences before moving on to longer ones both reduces overall training time and, surprisingly, substantially improves perplexity." (Abstract).

## 8. Positional Encoding (Critical Section)
- Mechanism used (absolute, sinusoidal): "uses sinusoidal position embeddings." (Section 2, Experimental Setup)
- Where applied in baseline: "Add the position embedding of each index to the token at that index: X = X + P." (Section 5, Transformer Language Models)
- Where applied in PIA: "model so that it does not add position embeddings at the *beginning* of the computation (step 2), but rather adds them to the query and key vectors at each layer (but *not* to the value vectors)." (Section 5.1)
- Fixed vs modified across experiments: "We use sinusoidal position embeddings; learned position embeddings, which we do not consider." (Section 4.1); PIA explicitly changes where embeddings are applied (Section 5.1).

## 9. Positional Encoding as a Variable
- Core research variable: Yes; "model so that it does not add position embeddings at the *beginning* of the computation (step 2), but rather adds them to the query and key vectors at each layer (but *not* to the value vectors)." (Section 5.1)
- Multiple positional encodings compared: Baseline absolute added to word embeddings vs PIA absolute added to queries/keys, plus mention of relative in prior work: "TransformerXL (Dai et al., 2019) uses *relative* position embeddings to solve this problem." (Section 5)
- PE choice claimed "not critical" or secondary: Not stated.

## 10. Evidence of Constraint Masking
- Model size(s): "The B&A model has 16 transformer layers of dimension 1,024, with 8 heads in each self-attention sublayer, and feedforward sublayers with an inner dimension of 4,096." (Section 2, Experimental Setup)
- Dataset size(s): "The training set contains 103.2 million tokens from English Wikipedia." (Section 2, Experimental Setup); "a 700M token collection of books" (A.5 Toronto Book Corpus)
- Performance gains attributed to training tricks/architecture rather than scaling: "Combining these techniques speeds up training by a factor of 1.65, reduces memory usage, and substantially improves perplexity on WikiText-103, without adding any parameters." (Abstract); "We do not change any hyperparameters other than reducing subsequence length while correspondingly increasing batch size to keep the number of tokens per batch constant." (Section 4.2)

## 11. Architectural Workarounds
- Staged training (short then long subsequences): "initially training on shorter subsequences (before moving to longer ones) leads not only to much faster and more memory-efficient training, but it surprisingly also greatly improves perplexity" (Section 1).
- Position-Infused Attention (PIA): "model so that it does not add position embeddings at the *beginning* of the computation (step 2), but rather adds them to the query and key vectors at each layer (but *not* to the value vectors)." (Section 5.1)
- Caching for recurrence: "Therefore, all our PIA models use a cache, where representations from the previous forward pass are stored and attended to in the next forward pass." (Section 5.2)
- Sliding window inference: "Here, we choose a stride S between 1 and L-1 and advance the window by S tokens after each forward pass." (Section 2, Sliding Window Inference)

## 12. Explicit Limitations and Non-Claims
- "In this paper we do not consider open-ended generation; we generate the dev. set, and for next-token prediction we use the ground truth token." (Section 2, Evaluation vs. Generation)
- "Therefore, it would not be applicable to sequence-to-sequence tasks such as sentence-level translation, where sequence lengths are short." (Section 5.2)
- "We use sinusoidal position embeddings; learned position embeddings, which we do not consider." (Section 4.1)

### 13. Constraint Profile (Synthesis)
**Constraint Profile:**
- Domain scope: natural language text only; Wikipedia and book corpus datasets, no multimodal data.
- Task structure: single task (language modeling/next-token prediction) with evaluation and generation modes.
- Representation rigidity: fixed-length token subsequences L per model (varied across experiments) and fixed cache length L'=L.
- Model sharing vs specialization: models are trained per dataset; no joint multi-task training stated.
- Role of positional encoding: core variable (baseline absolute sinusoidal vs PIA placement on queries/keys).

### 14. Final Classification
**Single-task, single-domain**

The paper evaluates a single task, language modeling, described as "In evaluation, a model assigns a perplexity score to a given sequence" and "In generation, a model generates a new sequence" (Section 2). The experiments are confined to text corpora ("trained and evaluated on WikiText-103" and "we ran our models on the Toronto Book Corpus (TBC) (Zhu et al., 2015)"), with no joint multi-task or multi-domain training described.
