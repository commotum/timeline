## 1. Basic Metadata

- Title: "Zero-Shot Text-to-Image Generation" (Title)
- Authors: "Aditya Ramesh <sup>1</sup> Mikhail Pavlov <sup>1</sup> Gabriel Goh <sup>1</sup> Scott Gray <sup>1</sup> Chelsea Voss <sup>1</sup> Alec Radford <sup>1</sup> Mark Chen <sup>1</sup> Ilya Sutskever <sup>1</sup>" (Title)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.


## 2. One-Sentence Contribution Summary

The paper proposes a transformer approach to text-to-image generation, stating that it is "a simple approach for this task based on a transformer that autoregressively models the text and image tokens as a single stream of data" (Abstract) to achieve zero-shot generation performance.


## 3. Tasks Evaluated

- Task name: Text-to-image generation (zero-shot)
  - Task type: Generation
  - Dataset(s) used: MS-COCO; CUB
  - Domain: natural images (caption-to-image)
  - Quotes: "The resulting system achieves high quality image generation on the popular MS-COCO dataset zero-shot, without using any of the training labels." (Introduction) ; "Figure 9. Quantitative results on MS-COCO and CUB." (Figure 9) ; "Our model fares significantly worse on the CUB dataset, for which there is a nearly 40-point gap in FID between our model and the leading prior approach" (Section 3.1).

- Task name: Zero-shot image-to-image translation (natural-language controlled)
  - Task type: Generation; Other (image-to-image translation)
  - Dataset(s) used: Not specified.
  - Domain: natural images
  - Quotes: "To a limited degree of reliability, we also find our model to be capable of zero-shot image-to-image translation controllable by natural language (Figure 2d)." (Section 3.3) ; "When the model is given the caption \"the exact same cat on the top as a sketch at the bottom\" and the top  $15 \times 32$  part of the image token grid for a photo of a cat, it is able to draw a sketch of a similar looking cat on the bottom." (Section 3.3).


## 4. Domain and Modality Scope

- Evaluation performed on a single domain? Multiple datasets within the same modality are evaluated: "MS-COCO" and "CUB" (Figure 9), both natural-image caption datasets; "The resulting system achieves high quality image generation on the popular MS-COCO dataset zero-shot" (Introduction) and "Our model fares significantly worse on the CUB dataset" (Section 3.1).
- Multiple domains within the same modality? Yes, MS-COCO vs. CUB (Figure 9; Section 3.1).
- Multiple modalities? The model jointly uses text and image tokens: "autoregressively models the text and image tokens as a single stream of data." (Abstract)
- Domain generalization or cross-domain transfer claim? Not claimed; the paper notes poorer zero-shot performance on a specialized distribution: "We speculate that our zero-shot approach is less likely to compare favorably on specialized distributions such as CUB." (Section 3.1).


## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Text-to-image generation (MS-COCO, CUB) | Yes (single model, zero-shot) | No (zero-shot) | Not specified. | "The resulting system achieves high quality image generation on the popular MS-COCO dataset zero-shot, without using any of the training labels." (Introduction) ; "Our model fares significantly worse on the CUB dataset" (Section 3.1). |
| Zero-shot image-to-image translation | Yes (same model, zero-shot) | No (zero-shot) | Not specified. | "we also find our model to be capable of zero-shot image-to-image translation controllable by natural language" (Section 3.3). |


## 6. Input and Representation Constraints

- Fixed image resolution and token grid: "We train a discrete variational autoencoder  $(dVAE)^1$  to compress each  $256 \times 256$  RGB image into a  $32 \times 32$  grid of image tokens" (Section 2).
- Fixed number of image tokens: "the  $32 \times 32 = 1024$  image tokens" (Section 2).
- Text length limit: "We concatenate up to 256 BPE-encoded text tokens" (Section 2) and "We limit the length of a text caption to 256 tokens" (Section 2.2).
- Padding for text positions: "we opt to learn a special padding token separately for each of the 256 text positions. This token is used only when no text token is available." (Section 2.2).
- Resizing/cropping requirements: "We use target_res = 256 and channel_count = 3." (Listing 1) ; "We also discard instances whose images have aspect ratios not in [1/2, 2]. If we were to use to very tall or wide images, then the square crops used during training would likely exclude objects mentioned in the caption." (Section C).
- Fixed downsampling factor: "The encoder downsamples the spatial resolution by a factor of 8." (Figure 1).


## 7. Context Window and Attention Structure

- Maximum sequence length: Not explicitly stated; the model concatenates "up to 256 BPE-encoded text tokens" with "the  $32 \times 32 = 1024$  image tokens" (Section 2).
- Fixed or variable length: Text length is capped and padded: "We limit the length of a text caption to 256 tokens" and "we opt to learn a special padding token separately for each of the 256 text positions" (Section 2.2); image tokens are a fixed  $32 \times 32$  grid (Section 2).
- Attention type: Sparse and structured attention masks are used: "a 12-billion parameter sparse transformer" (Section 2.2) with "row, column, or convolutional attention mask" (Section 2.2); cross-modal attention is global from image tokens to text tokens: "each image token can attend to all text tokens in any one of its 64 self-attention layers" (Section 2.2).
- Mechanisms to manage computational cost: Token compression to reduce context size: "compress each  $256 \times 256$  RGB image into a  $32 \times 32$  grid of image tokens" and "This reduces the context size of the transformer by a factor of 192" (Section 2).


## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: Row/column (axial) embeddings for image tokens: "broadcasted row and column embeddings for the part of the context for the image tokens" (Appendix B.1) and "Each image vocabulary embedding is summed with a row and column embedding." (Figure 10).
- Where it is applied: Summed into the image token embeddings at input: "Each image vocabulary embedding is summed with a row and column embedding." (Figure 10).
- Fixed across experiments / modified per task / ablated: Not specified; no ablations or alternatives are described.
- Text positional encoding: Not explicitly described beyond position-specific padding tokens: "we opt to learn a special padding token separately for each of the 256 text positions." (Section 2.2).


## 9. Positional Encoding as a Variable

- Positional encoding is not treated as a core research variable; no comparisons or ablations are described. Not specified.


## 10. Evidence of Constraint Masking

- Model size(s): "a 12-billion parameter sparse transformer" (Section 2.2).
- Dataset size(s): "Conceptual Captions, a dataset of 3.3 million text-image pairs" and "collecting 250 million text-images pairs from the internet" (Section 2.3).
- Scaling claims: "With sufficient data and scale, our approach is competitive with previous domain-specific models when evaluated in a zero-shot fashion." (Abstract) ; "training a 12-billion parameter autoregressive transformer on 250 million image-text pairs collected from the internet results in a flexible, high fidelity generative model" (Introduction) ; "We investigate a simple approach for text-to-image generation based on an autoregressive transformer, when it is executed at scale. We find that scale can lead to improved generalization" (Conclusion).
- Training tricks and evaluation procedures: "we rerank the samples drawn from the transformer using a pretrained contrastive model" (Section 2.6) and "Figure 9(c) shows clear improvements in FID and IS for MS-COCO as the sample size used for reranking with the contrastive model is increased." (Section 3.1).


## 11. Architectural Workarounds

- Two-stage compression to reduce context size: "We address these issues by using a two-stage training procedure" with "compress each  $256 \times 256$  RGB image into a  $32 \times 32$  grid of image tokens" and "This reduces the context size of the transformer by a factor of 192" (Section 2).
- Sparse attention masks: "There are three different kinds of self-attention masks used in the model" including "row, column, or convolutional attention mask" (Section 2.2).
- Fixed grid assumptions: "the  $32 \times 32 = 1024$  image tokens" (Section 2).
- Padding tokens for text positions: "we opt to learn a special padding token separately for each of the 256 text positions." (Section 2.2).
- Sample reranking: "we rerank the samples drawn from the transformer using a pretrained contrastive model" (Section 2.6).


## 12. Explicit Limitations and Non-Claims

- Compression limitation: "However, it also disadvantages the model, since the heavy compression renders it unable to produce high-frequency details." (Section 3.1).
- Dataset limitation: "Our model fares significantly worse on the CUB dataset, for which there is a nearly 40-point gap in FID between our model and the leading prior approach" (Section 3.1).
- Future work / fine-tuning: "We speculate that our zero-shot approach is less likely to compare favorably on specialized distributions such as CUB. We believe that fine-tuning is a promising direction for improvement, and leave this investigation to future work." (Section 3.1).
- Reliability limitation for image-to-image translation: "To a limited degree of reliability, we also find our model to be capable of zero-shot image-to-image translation" (Section 3.3).
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not stated.


### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Natural-image datasets (MS-COCO, CUB) with text-image pairs; same modality.
> - Task structure: Primarily text-to-image generation plus limited zero-shot image-to-image translation.
> - Representation rigidity: Fixed 256x256 inputs compressed to a 32x32 token grid; text capped at 256 tokens with padding.
> - Model sharing vs specialization: Single model evaluated zero-shot across tasks/datasets; no per-task fine-tuning reported.
> - Role of positional encoding: Row/column embeddings summed with image tokens; no ablations or variants discussed.


### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates text-to-image generation on MS-COCO and CUB (e.g., "The resulting system achieves high quality image generation on the popular MS-COCO dataset zero-shot" (Introduction) and "Our model fares significantly worse on the CUB dataset" (Section 3.1)) and also reports "zero-shot image-to-image translation" (Section 3.3), so more than one task is examined. All evaluations remain within natural images paired with text, so the scope stays single-domain.
