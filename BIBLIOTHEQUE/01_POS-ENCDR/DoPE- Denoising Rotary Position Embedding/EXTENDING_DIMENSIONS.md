## 1. Basic Metadata
- Title: "DoPE: Denoising Rotary Position Embedding" (Title)
- Authors: "Jing Xiong<sup>1\*</sup>, Liyang Fan<sup>3\*</sup>, Hui Shen<sup>2</sup>, Zunhai Su<sup>1</sup>, Min Yang<sup>3†</sup>, Lingpeng Kong<sup>1</sup>, and Ngai Wong<sup>1</sup>" (Front matter)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary
Proposes DoPE, a training-free denoising positional encoding method to address RoPE length extrapolation limits and attention sinks ("Rotary Position Embedding (RoPE) in Transformer models has inherent limits that weaken length extrapolation. We reinterpret the attention map with positional encoding as a noisy feature map, and propose Denoising Positional Encoding (DoPE), a training-free method based on truncated matrix entropy to detect outlier frequency bands in the feature map." Abstract).

## 3. Tasks Evaluated

Task 1:
- Task name: Needle-in-a-haystack (NIH)
- Task type: Other (specify: retrieval)
- Dataset(s) used: Not specified.
- Domain: natural language processing / information retrieval.
- Evidence: 'The "needle-in-a-haystack" synthesis task presents a particularly challenging problem in the field of natural language processing and information retrieval.' (Section 5.1 Experimental Setup); "Experiments on needle-in-a-haystack and many-shot in-context learning tasks demonstrate that DoPE significantly improves retrieval accuracy and reasoning stability across extended contexts (up to 64K tokens)." (Abstract)

Task 2:
- Task name: Many-shot in-context learning (MICL) / in-context learning
- Task type: Reasoning / relational
- Dataset(s) used: "The MICL task evaluates 100 sampled problems from the MATH dataset (Hendrycks et al., 2021), with needle insertion at four fixed depth positions (0%, 33%, 67%, 100%, corresponding to beginning, 1/3, 2/3, and end) within the in-context examples, yielding 400 total test configurations." (Appendix A.2 Experimental Setup)
- Domain: natural language / math text.
- Evidence: "We present the model's performance under many-shot in-context learning (MICL) scenarios (Agarwal et al., 2024) in Table 2. Experiments are conducted both with test exemplar inserted into the in-context exemplars (needle-in-a-haystack) and without test exemplars (in-context learning)." (Section 5.3 Many-Shot In-Context Learning); "This task not only depends on the model's ability to find a needle in a haystack, but also tests whether the model can identify similar reasoning patterns from the context." (Section 5.3 Many-Shot In-Context Learning)

## 4. Domain and Modality Scope
- Evaluation domain/modality: Single domain, single modality (text/LLM). Evidence: "We consider a causal language model implemented as a decoder-only Transformer." (Section 2.1 Multi-Head Self-Attention); 'The "needle-in-a-haystack" synthesis task presents a particularly challenging problem in the field of natural language processing and information retrieval.' (Section 5.1 Experimental Setup)
- Multiple domains within the same modality? Not stated.
- Multiple modalities? Not stated.
- Domain generalization or cross-domain transfer? Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Needle-in-a-haystack (NIH) | No (task evaluated on LLaMA-3-8B-Instruct) | No (training-free) | Not specified | "For the needle-in-a-haystack (NIH) task on LLaMA-3-8B-Instruct," (Appendix A.2 Experimental Setup); "a training-free method" (Abstract) |
| Many-shot in-context learning (MICL) / in-context learning | No (task evaluated on Qwen2.5-Math-7B) | No (training-free) | Not specified | "For the many-shot in-context learning (MICL) task on Qwen2.5-Math-7B," (Appendix A.2 Experimental Setup); "a training-free method" (Abstract) |

## 6. Input and Representation Constraints
- Token-sequence inputs: "Given token representations  $\mathbf{X} \in \mathbb{R}^{n \times d}$ , we form queries, keys, and values via" (Section 2.1 Multi-Head Self-Attention).
- Autoregressive ordering: "Let  $\mathbf{M} \in \mathbb{R}^{1 \times n \times n}$  be the causal mask, with 0 on and below the diagonal and  $-\infty$  above." (Section 2.1 Multi-Head Self-Attention).
- Head dimension constraint: "Let the per-head width be  $d_h$  (assume  $d_h$  is even)." (Section 2.2 Rotary Position Embedding).
- Training/evaluation context lengths: "Qwen-1.5-7B is trained with a maximum context length of 32K tokens, while LLaMA-3-8B is trained with a 8K-token context window." (Appendix A.2 Experimental Setup); "where  $L_{\text{target}} \in \{24\text{K}, 64\text{K}, 128\text{K}\}$  for NIH experiments and  $L_{\text{target}} = 16\text{K}$  for MICL experiments" (Appendix A.2 Experimental Setup).
- Training-length-dependent masking: "$$\theta = \frac{2\pi}{L},\tag{33}$$

where L is the training length." (Section 4.3 Denoising via Truncated Matrix Entropy).
- Input resolution / patch size / padding or resizing requirements: Not specified.

## 7. Context Window and Attention Structure
- Maximum sequence length: "where  $L_{\text{target}} \in \{24\text{K}, 64\text{K}, 128\text{K}\}$  for NIH experiments and  $L_{\text{target}} = 16\text{K}$  for MICL experiments" (Appendix A.2 Experimental Setup); "across extended contexts (up to 64K tokens)." (Abstract)
- Fixed or variable length: "CUDA graphs are disabled to support dynamic context lengths." (Section 5.2 Main results)
- Attention type: Global causal self-attention ("Let  $\mathbf{M} \in \mathbb{R}^{1 \times n \times n}$  be the causal mask, with 0 on and below the diagonal and  $-\infty$  above." and "$$\mathbf{A} = \operatorname{softmax} \left( \frac{\mathbf{Q} \mathbf{K}^{\top}}{\sqrt{d_h}} + \mathbf{M} \right) \in \mathbb{R}^{h \times n \times n}, \quad (6)$$" Section 2.1 Multi-Head Self-Attention).
- Mechanisms to manage computational cost (windowing/pooling/pruning): Not stated.

## 8. Positional Encoding (Critical Section)
- Mechanism: RoPE ("Most LLMs adopt *Rotary Position Embedding* (RoPE) (Su et al., 2024) as their default positional encoding mechanism, which has become the de facto standard in contemporary architectures. RoPE encodes token positions by rotating each query/key vector on a sequence of two-dimensional planes." Section 2.2 Rotary Position Embedding).
- Where applied: to query and key vectors ("Position encodings are often added to the query and key vectors to incorporate sequence order." Introduction).
- Application across layers: Not specified.
- Fixed vs modified / ablated: Modified and compared across RoPE variants and DoPE strategies ("We reinterpret the attention map with positional encoding as a noisy feature map, and propose Denoising Positional Encoding (DoPE), a training-free method based on truncated matrix entropy to detect outlier frequency bands in the feature map." Abstract; "For RoPE extrapolation, we apply Dynamic-NTK scaling (emoZilla, 2023) with the scaling factor computed as  $\alpha = L_{\text{target}}/L_{\text{original}}$ , where  $L_{\text{target}} \in \{24\text{K}, 64\text{K}, 128\text{K}\}$  for NIH experiments and  $L_{\text{target}} = 16\text{K}$  for MICL experiments, while  $L_{\text{original}}$  corresponds to each model's pre-trained maximum position embeddings (32K for Qwen-1.5-7B, 8K for LLaMA-3-8B-Instruct, and 4K for Qwen2.5-Math-7B)." Appendix A.2 Experimental Setup; "For LLaMA-3, we additionally evaluate NTK-by-parts (Peng et al., 2023) with low_freq_factor= 1.0 and high_freq_factor= 32.0." Appendix A.2 Experimental Setup; "We also evaluate the full (untruncated) matrix entropy for comparison." Appendix A.2 Experimental Setup; "Only when  $m_h = 1$  do we remove the positional encoding for this head; otherwise, the RoPE positional encoding of that head remains unchanged (all bands are retained)." Section 4.3 Denoising via Truncated Matrix Entropy).

## 9. Positional Encoding as a Variable
- Core research variable? Yes: "We reinterpret the attention map with positional encoding as a noisy feature map, and propose Denoising Positional Encoding (DoPE), a training-free method based on truncated matrix entropy to detect outlier frequency bands in the feature map." (Abstract)
- Fixed architectural assumption? No; multiple variants are tested ("For RoPE extrapolation, we apply Dynamic-NTK scaling (emoZilla, 2023) with the scaling factor computed as  $\alpha = L_{\text{target}}/L_{\text{original}}$ , where  $L_{\text{target}} \in \{24\text{K}, 64\text{K}, 128\text{K}\}$  for NIH experiments and  $L_{\text{target}} = 16\text{K}$  for MICL experiments, while  $L_{\text{original}}$  corresponds to each model's pre-trained maximum position embeddings (32K for Qwen-1.5-7B, 8K for LLaMA-3-8B-Instruct, and 4K for Qwen2.5-Math-7B)." Appendix A.2 Experimental Setup; "For LLaMA-3, we additionally evaluate NTK-by-parts (Peng et al., 2023) with low_freq_factor= 1.0 and high_freq_factor= 32.0." Appendix A.2 Experimental Setup; "We also evaluate the full (untruncated) matrix entropy for comparison." Appendix A.2 Experimental Setup).
- Multiple positional encodings compared? Yes (Dynamic-NTK, NTK-by-parts, truncated vs full matrix entropy) with the same evidence above.
- Claim PE choice is "not critical" or secondary? Not stated.

## 10. Evidence of Constraint Masking
- Model size(s): "Qwen-1.5-7B (Li, 2023), Qwen2.5-Math-7B (Yang et al., 2024) and LLaMA-3-8B-Instruct (Grattafiori et al., 2024) are decoder-only transformer models that employ Rotary Positional Embeddings (RoPE) for encoding positional information." (Appendix A.2 Experimental Setup)
- Dataset size(s) / evaluation scale: "The MICL task evaluates 100 sampled problems from the MATH dataset (Hendrycks et al., 2021), with needle insertion at four fixed depth positions (0%, 33%, 67%, 100%, corresponding to beginning, 1/3, 2/3, and end) within the in-context examples, yielding 400 total test configurations." (Appendix A.2 Experimental Setup); "The NIH task uses 10 uniformly spaced depth positions (0%, 10%, ..., 100%) for needle insertion at each context length." (Appendix A.2 Experimental Setup)
- Attribution of gains: "The results show that the denoising strategy for positional embeddings effectively mitigates attention sinks and restores balanced attention patterns, providing a simple yet powerful solution for improving length generalization." (Abstract); "Experiments on needle-in-a-haystack and many-shot in-context learning tasks demonstrate that DoPE significantly improves retrieval accuracy and reasoning stability across extended contexts (up to 64K tokens)." (Abstract)
- Scaling model size or data? Not claimed; method is training-free ("We reinterpret the attention map with positional encoding as a noisy feature map, and propose Denoising Positional Encoding (DoPE), a training-free method based on truncated matrix entropy to detect outlier frequency bands in the feature map." Abstract).

## 11. Architectural Workarounds
- RoPE extrapolation via frequency rescaling: "To support longer contexts beyond their pre-training limits, we apply RoPE-based extrapolation (e.g., Dynamic-NTK), which rescales RoPE frequencies to improve stability and retrieval performance in extended-context settings." (Appendix A.2 Experimental Setup)
- NTK-by-parts variant for extrapolation: "For LLaMA-3, we additionally evaluate NTK-by-parts (Peng et al., 2023) with low_freq_factor= 1.0 and high_freq_factor= 32.0." (Appendix A.2 Experimental Setup)
- Head-level positional encoding masking: "Only when  $m_h = 1$  do we remove the positional encoding for this head; otherwise, the RoPE positional encoding of that head remains unchanged (all bands are retained)." (Section 4.3 Denoising via Truncated Matrix Entropy)
- Frequency band masking tied to training length: "Formally, for the selected head h, we construct a *frequency band mask* based on the threshold:" and "$$\theta = \frac{2\pi}{L},\tag{33}$$

where L is the training length." (Section 4.3 Denoising via Truncated Matrix Entropy)
- DoPE-by-all head-level masking: "In this variant, denoising is performed by applying the head-level mask to the entire positional encoding of each head, rather than completely zeroing out the head." (Section 4.3 Denoising via Truncated Matrix Entropy)
- Gaussian noise reparameterization: "Leveraging the noise characteristics of the feature map, we further reparameterize it with a parameter-free Gaussian distribution to achieve robust extrapolation." (Abstract); "For DoPE, Gaussian noise is sampled from  $\mathcal{N}(0,1)$  with standard deviation  $\sigma = 1.0$ , using a fixed random seed (42) to ensure reproducibility." (Appendix A.2 Experimental Setup)

## 12. Explicit Limitations and Non-Claims
- Limitation on long contexts: "The Curse of Length. At an appropriate length, MICL can significantly enhance the model's reasoning ability. However, when the length extends to 16K, the model's final reasoning ability drops significantly. More exemplars does not lead to better performance, indirectly demonstrating that complex reasoning is constrained by the extrapolation length." (Section 5.3 Many-Shot In-Context Learning)
- Limitation with needle insertion at long lengths: "The Curse of Shortcut. We inserted exemplars of the test samples into the in-context examples. Surprisingly, rather than copying the correct answers in a \"needle-in-a-haystack\" manner, the model's overall performance dropped substantially at the 24K and 64K context lengths." (Section 5.3 Many-Shot In-Context Learning)
- Explicit non-claims (open-world learning, unrestrained multi-task learning, meta-learning): Not stated.

### 13. Constraint Profile (Synthesis)
> **Constraint Profile:**
> - Domain scope: Single text/NLP domain ("We consider a causal language model implemented as a decoder-only Transformer." Section 2.1 Multi-Head Self-Attention; 'The "needle-in-a-haystack" synthesis task presents a particularly challenging problem in the field of natural language processing and information retrieval.' Section 5.1 Experimental Setup).
> - Task structure: Two explicit evaluated tasks with in-context variants ("Experiments on needle-in-a-haystack and many-shot in-context learning tasks demonstrate that DoPE significantly improves retrieval accuracy and reasoning stability across extended contexts (up to 64K tokens)." Abstract; "Experiments are conducted both with test exemplar inserted into the in-context exemplars (needle-in-a-haystack) and without test exemplars (in-context learning)." Section 5.3 Many-Shot In-Context Learning).
> - Representation rigidity: Token sequence modeling with fixed context lengths per experiment and training-length-dependent masking ("Qwen-1.5-7B is trained with a maximum context length of 32K tokens, while LLaMA-3-8B is trained with a 8K-token context window." Appendix A.2 Experimental Setup; "$$\theta = \frac{2\pi}{L},\tag{33}$$\n\nwhere L is the training length." Section 4.3 Denoising via Truncated Matrix Entropy).
> - Model sharing vs specialization: Different base models per task and no fine-tuning ("For the needle-in-a-haystack (NIH) task on LLaMA-3-8B-Instruct," Appendix A.2 Experimental Setup; "For the many-shot in-context learning (MICL) task on Qwen2.5-Math-7B," Appendix A.2 Experimental Setup; "a training-free method" Abstract).
> - Role of positional encoding: Central variable with multiple RoPE variants and denoising ("We reinterpret the attention map with positional encoding as a noisy feature map, and propose Denoising Positional Encoding (DoPE), a training-free method based on truncated matrix entropy to detect outlier frequency bands in the feature map." Abstract; "For RoPE extrapolation, we apply Dynamic-NTK scaling (emoZilla, 2023) with the scaling factor computed as  $\alpha = L_{\text{target}}/L_{\text{original}}$ , where  $L_{\text{target}} \in \{24\text{K}, 64\text{K}, 128\text{K}\}$  for NIH experiments and  $L_{\text{target}} = 16\text{K}$  for MICL experiments, while  $L_{\text{original}}$  corresponds to each model's pre-trained maximum position embeddings (32K for Qwen-1.5-7B, 8K for LLaMA-3-8B-Instruct, and 4K for Qwen2.5-Math-7B)." Appendix A.2 Experimental Setup; "Only when  $m_h = 1$  do we remove the positional encoding for this head; otherwise, the RoPE positional encoding of that head remains unchanged (all bands are retained)." Section 4.3 Denoising via Truncated Matrix Entropy).

### 14. Final Classification
**Multi-task, single-domain**

The paper evaluates more than one task in language modeling, explicitly "needle-in-a-haystack" and "many-shot in-context learning" ("Experiments on needle-in-a-haystack and many-shot in-context learning tasks demonstrate that DoPE significantly improves retrieval accuracy and reasoning stability across extended contexts (up to 64K tokens)." Abstract; "We present the model's performance under many-shot in-context learning (MICL) scenarios (Agarwal et al., 2024) in Table 2." Section 5.3 Many-Shot In-Context Learning). The evaluation remains within a single text/NLP domain using causal language models ("We consider a causal language model implemented as a decoder-only Transformer." Section 2.1 Multi-Head Self-Attention; 'The "needle-in-a-haystack" synthesis task presents a particularly challenging problem in the field of natural language processing and information retrieval.' Section 5.1 Experimental Setup), with no cross-domain or multi-modal transfer claims.
