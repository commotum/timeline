## 1. Basic Metadata

- Title: "A Length-Extrapolatable Transformer" (Title header)
- Authors: "Yutao Sun, Li Dong, Barun Patra, Shuming Ma Shaohan Huang, Alon Benhaim, Vishrav Chaudhary, Xia Song, Furu Wei Microsoft" (Title header)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

In the Abstract, the paper states "we focus on length extrapolation, i.e., training on short texts while evaluating longer sequences," "we introduce a relative position embedding to explicitly maximize attention resolution," "we use blockwise causal attention during inference for better resolution," and "We evaluate different Transformer variants with language modeling." (Abstract)

## 3. Tasks Evaluated

Task 1: Language modeling

- Task name: Language modeling.
- Task type: Generation.
- Dataset(s) used: "We use the arXiv dataset (above 6k length) to evaluate the model's ability for extrapolation length." (1 Introduction); "We first measure perplexity on arXiv, where the document length is usually larger than 6k..." (4.2 Language Modeling); "The training corpus includes a subset of the Pile (Gao et al., 2020): Books3, OpenWebText2, Stack Exchange, PubMed Abstracts, Wikipedia, Gutenberg (PG-19), BookCorpus2, NIH ExPorter, and Pile-CC datasets." (Pre-training); "After pre-training, we test the perplexity on the valid split of training corpus with 1k length." (4.4.1 Rotation Computation).
- Domain: Text (language modeling); evidence: "We evaluate different Transformer variants with language modeling." (Abstract); "In this work, we focus on causal language modeling." (Limitations).
- Task description quotes: "We evaluate different Transformer variants with language modeling." (Abstract); "We conduct experiments on language modeling and show that the proposed LEX Transformer achieves strong performance on both short and long texts." (Contributions).

## 4. Domain and Modality Scope

- Evaluation domain/modality: Single modality (text) with language modeling; evidence: "We evaluate different Transformer variants with language modeling." (Abstract); "In this work, we focus on causal language modeling." (Limitations).
- Single domain vs multiple domains within the same modality: Multiple text domains within the same modality across datasets (arXiv plus Pile subset); evidence: "We use the arXiv dataset (above 6k length) to evaluate the model's ability for extrapolation length." (1 Introduction); "The training corpus includes a subset of the Pile (Gao et al., 2020): Books3, OpenWebText2, Stack Exchange, PubMed Abstracts, Wikipedia, Gutenberg (PG-19), BookCorpus2, NIH ExPorter, and Pile-CC datasets." (Pre-training).
- Multiple modalities: Not indicated; no evidence of non-text modalities in the evaluation.
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Language modeling | N/A (single task; same model evaluated at multiple lengths) | Not specified; trained from scratch | Not specified | "We pre-train the Transformer from scratch." (Pre-training); "The language models are trained with a length of 1024 and then evaluated on various lengths." (Table 2) |

## 6. Input and Representation Constraints

- Fixed or variable input resolution/length: "Maximal length is 1024 for saving memory and extrapolation evaluation." (Pre-training); "For every document, we select its first 4k tokens and divide them into the target length to fairly compare the perplexity of different lengths." (4.2 Language Modeling).
- Fixed patch size: Not specified.
- Fixed number of tokens: "The language models are trained with a length of 1024 and then evaluated on various lengths." (Table 2).
- Fixed dimensionality (e.g., strictly 2D): Not specified.
- Padding or resizing requirements: "For instance, in fact, a sentence's meaning is variant with padding before or after the whole sentence." (2.2 Translation Invariance).

## 7. Context Window and Attention Structure

- Maximum sequence length: "Maximal length is 1024 for saving memory and extrapolation evaluation." (Pre-training); "For every document, we select its first 4k tokens and divide them into the target length to fairly compare the perplexity of different lengths." (4.2 Language Modeling).
- Fixed vs variable sequence length: "The language models are trained with a length of 1024 and then evaluated on various lengths." (Table 2).
- Attention type: "we use vanilla attention in the training phase" (Blockwise Causal Attention 3.3); "During inference, we use blockwise masking (Dai et al., 2019; Zaheer et al., 2020; Xiong et al., 2021) for selfattention." (Blockwise Causal Attention 3.3).
- Mechanisms to manage computational cost or long-context handling: "We use blockwise causal attention because it is cache-friendly and easy to implement." (Blockwise Causal Attention 3.3); "The window constraint helps models to encode longer input with improved resolution." (Blockwise Causal Attention 3.3); "If the pre-training length is l, we divide the query as blocks with l/2 length, and each query interacts with its own block and the last block." (Blockwise Causal Attention 3.3).

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: "we introduce a relative position embedding to explicitly maximize attention resolution." (Abstract); "Based on ROPE's design, we propose attention resolution as a metric to measure position monotonicity accurately." (1 Introduction); "Then, we generalize its mathematical form, where an exponential decay is added to the rotation matrix." (1 Introduction); "If xi=0, the form is the same as RoPE (Su et al., 2021)." (3.2 Improve Resolution by Position Encoding); "Finally, we have Extrapolatable Position Embedding (XPOS)." (3.2 Improve Resolution by Position Encoding).
- Where it is applied: "by adding absolute position embedding on query and key, the attention matrix is actually encoded with relative position information." (3.2 Improve Resolution by Position Encoding); "Q = (Q x C + rot(Q) x S) x T" and "K = (K x C + rot(K) x S) x T^{-1}" (Algorithm 1: Attention with XPOS).
- Fixed across experiments or modified/ablated: "We evaluate different Transformer variants with language modeling." (Abstract); "To fairly compare different methods, we run the evaluation using different position embeddings (i.e., Alibi, RoPE, and xPos) with or without blockwise causal attention." (4.4.2 Blockwise Causal Attention); "In this part, we discuss the necessity of the combination of vector rotation and exponential decay." (4.4.1 Rotation Computation).

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption: Core research variable; evidence: "We define attention resolution as the indicator of length extrapolation... we introduce a relative position encoding method (Section 3.2) to explicitly maximize attention resolution." (3 A Length-Extrapolatable Transformer).
- Multiple positional encodings compared: Yes; "We evaluate different Transformer variants with language modeling." (Abstract); "To fairly compare different methods, we run the evaluation using different position embeddings (i.e., Alibi, RoPE, and xPos)." (4.4.2 Blockwise Causal Attention).
- Claim that PE choice is not critical or secondary: Not claimed.

## 10. Evidence of Constraint Masking

- Model size(s): "We use 1024 hidden dimension, 16 heads, and 24 layers, i.e., comparable to medium-size GPT-3 (Brown et al., 2020)." (Pre-training).
- Dataset size(s): Dataset sizes not specified; training sources are listed: "The training corpus includes a subset of the Pile (Gao et al., 2020): Books3, OpenWebText2, Stack Exchange, PubMed Abstracts, Wikipedia, Gutenberg (PG-19), BookCorpus2, NIH ExPorter, and Pile-CC datasets." (Pre-training).
- Attributed source of gains: The paper attributes gains to position encoding and attention masking, not to scaling model or data size: "we introduce a relative position embedding to explicitly maximize attention resolution" and "we use blockwise causal attention during inference for better resolution." (Abstract); "The window constraint helps models to encode longer input with improved resolution." (Blockwise Causal Attention 3.3).

## 11. Architectural Workarounds

- Relative position embedding with decay (XPOS) to improve extrapolation: "we introduce a relative position embedding to explicitly maximize attention resolution." (Abstract); "Then, we generalize its mathematical form, where an exponential decay is added to the rotation matrix." (1 Introduction).
- Blockwise causal attention (windowed attention) during inference to handle long sequences and improve resolution: "During inference, we use blockwise masking... for selfattention." (Blockwise Causal Attention 3.3); "The window constraint helps models to encode longer input with improved resolution." (Blockwise Causal Attention 3.3); "If the pre-training length is l, we divide the query as blocks with l/2 length, and each query interacts with its own block and the last block." (Blockwise Causal Attention 3.3).
- Training uses standard causal masking; inference reuses cached context: "Our language model is trained on shorter texts in the same way as vanilla Transformers, i.e., using causal masking. During inference, we use blockwise causal attention for longer sequences, which recurrently reuses the overlapped parts (i.e., key and value vectors)." (Figure 2).

## 12. Explicit Limitations and Non-Claims

- "In this work, we focus on causal language modeling. It needs additional efforts to integrate the proposed methods into bidirectional attention, such as masked language modeling (Devlin et al., 2019)." (Limitations).
- "Moreover, xPos introduces about 6% inference cost compared with absolute position embeddings, although it accelerates training convergence." (Limitations).
- Other explicit non-claims (e.g., open-world learning, unrestrained multi-task learning, meta-learning): Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Text-only language modeling on arXiv and Pile-subset corpora ("language modeling"; "arXiv"; "subset of the Pile").
> - Task structure: Single task focused on perplexity across lengths ("We evaluate different Transformer variants with language modeling.").
> - Representation rigidity: Training length fixed at 1024 with evaluation at multiple lengths ("Maximal length is 1024"; "evaluated on various lengths").
> - Model sharing vs specialization: Same model trained from scratch and evaluated across lengths; no task-specific heads mentioned ("We pre-train the Transformer from scratch.").
> - Role of positional encoding: Central variable with explicit comparisons/ablations ("we introduce a relative position embedding"; "different position embeddings (i.e., Alibi, RoPE, and xPos)").

### 14. Final Classification

**Single-task, single-domain.** The paper evaluates a single task, stating "We evaluate different Transformer variants with language modeling" and presenting "4.2 Language Modeling." (Abstract; 4.2 Language Modeling). Evaluation is text-only on datasets such as "arXiv" and a "subset of the Pile," with no evidence of additional modalities or distinct task heads. (1 Introduction; Pre-training)
