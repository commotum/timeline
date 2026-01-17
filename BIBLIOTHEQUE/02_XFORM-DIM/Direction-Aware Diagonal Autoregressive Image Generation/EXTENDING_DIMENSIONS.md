## 1. Basic Metadata
- Title: Direction-Aware Diagonal Autoregressive Image Generation
- Authors: Yijia Xu; Jianzhong Ju; Jian Luan; Jinshi Cui
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary
The paper proposes a "Direction-Aware Diagonal Autoregressive Image Generation (DAR) method, which generates image tokens following a diagonal scanning order" and introduces "4D-RoPE and direction embeddings" to handle frequent direction changes in autoregressive image generation (Abstract).

## 3. Tasks Evaluated
- Task: Class-conditional image generation; Task type: Generation; Dataset(s): 256×256 ImageNet-1K; Domain: ImageNet images (not further specified); Evidence: "In Tab. 2, we compare DAR with other image generation methods on class-conditional image generation task." (Section 4.2 Main Results) "Table 2. **256**×**256 ImageNet class-conditional generation results evaluated with ADM [10].**" (Table 2 caption) "We train our model on the 256×256 ImageNet-1K [8] dataset, which comprises 1,000 classes and a total of 1,281,167 images." (Section 4.1 Implementations Details)

## 4. Domain and Modality Scope
- Single domain: Yes; "We train our model on the 256×256 ImageNet-1K [8] dataset, which comprises 1,000 classes and a total of 1,281,167 images." (Section 4.1 Implementations Details)
- Multiple domains within the same modality: Not stated; evaluation is reported as "**256**×**256 ImageNet class-conditional generation results" (Table 2 caption).
- Multiple modalities: Not evaluated; the reported task is "image generation" on ImageNet (Section 4.2 Main Results).
- Domain generalization or cross-domain transfer: Not claimed; "Our enhanced autoregressive transformer is also compatible with unified multimodal generative models. We leave these for future work." (D. Limitations and Future Work)

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| ImageNet class-conditional image generation | N/A (single task) | Not specified | Not specified | "We train our model on the 256×256 ImageNet-1K [8] dataset" (Section 4.1 Implementations Details); "class-conditional image generation task" (Section 4.2 Main Results) |

## 6. Input and Representation Constraints
- 2D RGB image input and spatial coordinates: "An input image  $\mathcal{I} \in \mathbb{R}^{H \times W \times 3}$" and "discrete image token sequences produced by visual tokenizers maintain two-dimensional spatial coordinates." (Section 3.1 Preliminary; Introduction)
- Downsampled token grid and sequence length: "where h = H/p, w = W/p, p is the downsample ratio" and "sequence  $\mathbf{x} = [x_1, x_2, ..., x_T]$ , where  $T = h \times w$ ." (Section 3.1 Preliminary)
- Fixed resolution in experiments: "We train our model on the 256×256 ImageNet-1K [8] dataset" and "trained on  $256 \times 256$  ImageNet [8]" (Section 4.1 Implementations Details).
- Fixed token grid for 256×256 images: "it converts  $256 \times 256$  resolution images into  $16 \times 16$  discrete tokens." (Section 4.1 Implementations Details)
- Codebook size/dimensionality: "codebook size 16,384 and a code dimension 256." (Section 4.1 Implementations Details)
- Resizing noted for some baselines: "-384 indicates that images are initially generated at a resolution of  $384 \times 384$  and subsequently resized to  $256 \times 256$  for evaluation." (Table 2 caption)

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified.
- Sequence length fixed vs. variable: "sequence  $\mathbf{x} = [x_1, x_2, ..., x_T]$ , where  $T = h \times w$" (Section 3.1 Preliminary) and "it converts  $256 \times 256$  resolution images into  $16 \times 16$  discrete tokens" (Section 4.1 Implementations Details).
- Attention type: Causal attention; "Under the constraint of causal attention,  $x_{cur}$  can only attend to the preceding tokens when predicting  $x_{nxt}$ ." (Section 3.2 Diagonal Scanning Order)
- Computational cost mechanisms (windowing/pooling/token pruning): Not specified.

## 8. Positional Encoding (Critical Section)
- Mechanism: 4D-RoPE (rotary/relative); "we propose 4D-RoPE that incorporates both the current position and the next position" and "We refer to this method of using four-dimensional coordinates for RoPE as 4D-RoPE." (Introduction; Section 3.3 4D-RoPE and Direction Embeddings)
- Where applied: Attention matrix; "The variations in relative positions between any two tokens are injected into the attention matrix" and "After injecting the information of both  $\mathbf{p}_{cur}$  and  $\mathbf{p}_{nxt}$  into the attention matrix" (Introduction; Section 3.3 4D-RoPE and Direction Embeddings)
- Applied in model: "enhance the RoPE mechanism to a 4D-RoPE that combines both the current and the next positions." (Section 4.1 Implementations Details)
- Alternatives noted (prior work): "previous methods employ 2D-RoPE to inject 2D positional information of image tokens into the attention matrix" (Section 3.1 Preliminary)

## 9. Positional Encoding as a Variable
- Core research variable: Yes; "we conduct ablation studies on the image token scanning order and several key modules" and "we validate the effectiveness of our proposed two direction-aware modules: 4D-RoPE and direction embeddings." (Section 4.3 Ablation Studies)
- Multiple positional encodings compared: Only with/without 4D-RoPE is explicitly ablated; no direct comparison of multiple PE types is stated beyond noting prior 2D-RoPE (Section 4.3 Ablation Studies; Section 3.1 Preliminary).
- PE choice claimed as non-critical or secondary: Not stated.

## 10. Evidence of Constraint Masking
- Model scaling: "We propose models of varying scales, ranging from 485M to 2.0B." (Abstract) and "DAR demonstrates consistent performance improvements as the model scale increases (from 485M to 2.0B)." (Section 4.2 Main Results)
- Dataset size: "We train our model on the 256×256 ImageNet-1K [8] dataset, which comprises 1,000 classes and a total of 1,281,167 images." (Section 4.1 Implementations Details)
- Scaling behavior: "As the model size scales, subfigure (a) demonstrates a consistent reduction in loss, while subfigures (b) and (c) illustrate a consistent decrease in FID score" (Figure 4 caption).
- Architecture attribution: "These results demonstrate the effectiveness of our architectural optimizations for the autoregressive transformer." (Section 4.2 Main Results)
- Training/sampling tricks: "we employ classifier-free guidance [20] and the power-cosine guidance schedule [15], without utilizing top-k or top-p techniques." (Section 4.1 Implementations Details)

## 11. Architectural Workarounds
- Diagonal scanning order to keep neighbors close and widen directional context: "we rearrange the image tokens in diagonal scanning order, ensuring that all tokens with adjacent indices are positioned in close proximity" and "The proposed diagonal scanning order ensures that tokens with adjacent indices remain in close proximity while enabling causal attention to gather information from a broader range of directions." (Introduction; Abstract)
- Direction-aware positional modeling: "we propose 4D-RoPE that incorporates both the current position and the next position" to handle changing directions (Introduction), and "To explicitly condition the model on the generation direction, we propose 4D-RoPE" (Section 3.3 4D-RoPE and Direction Embeddings).
- Direction embeddings in AdaLN: "we use direction embeddings to directly represent generation directions and utilize them to calculate the scale and shift parameters in AdaLN" (Introduction) and "The direction embeddings are then summed with the class embedding to calculate the scale and shift parameters in AdaLN." (Section 3.3 4D-RoPE and Direction Embeddings)
- Codebook-based image token embeddings: "we directly utilize the codebook from the image tokenizer as the image token embeddings and freeze their parameters." (Section 3.4 Codebook-based Image Token Embeddings)
- Class embedding conditioning: "The class embedding is prepended to the sequence" and "AdaLN calculates the scale and shift parameters using the sum of class embeddings and direction embeddings." (Figure 3 caption)

## 12. Explicit Limitations and Non-Claims
- "In this work, we primarily focus on architectural improvements to the autoregressive transformer." (D. Limitations and Future Work)
- "We train our model on ImageNet [8], specifically focusing on class-conditional generation." (D. Limitations and Future Work)
- "The generative performance could be further enhanced by leveraging image-text paired data." (D. Limitations and Future Work)
- "Our enhanced autoregressive transformer is also compatible with unified multimodal generative models. We leave these for future work." (D. Limitations and Future Work)

### 13. Constraint Profile (Synthesis)
**Constraint Profile:**
- Domain scope: Single ImageNet-1K domain at 256×256 resolution; no cross-domain evaluation reported.
- Task structure: Class-conditional image generation only, evaluated on ImageNet.
- Representation rigidity: 2D RGB inputs tokenized to a fixed 16×16 grid for 256×256 images; sequence length tied to h×w.
- Model sharing vs specialization: Single-task training on ImageNet; no multi-task sharing or fine-tuning described.
- Role of positional encoding: 4D-RoPE is a core direction-aware module and is ablated for effectiveness.

### 14. Final Classification
**Single-task, single-domain.** The evaluation is limited to "class-conditional image generation task" on ImageNet, with results reported as "**256**×**256 ImageNet class-conditional generation results" (Section 4.2 Main Results; Table 2 caption). The authors also state they "train our model on ImageNet [8], specifically focusing on class-conditional generation," and no multi-domain or multi-task evaluation is presented (D. Limitations and Future Work).
