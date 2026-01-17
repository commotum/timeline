## 1. Basic Metadata

- Title: "Image Transformer" (Title)
- Authors: "Niki Parmar \*  $^1$  Ashish Vaswani \*  $^1$  Jakob Uszkoreit  $^1$  Łukasz Kaiser  $^1$  Noam Shazeer  $^1$  Alexander Ku  $^2$  3 Dustin Tran  $^4$" (Front matter, above Abstract)
- Year: "Proceedings of the 35<sup>th</sup> International Conference on Machine Learning, Stockholm, Sweden, PMLR 80, 2018." (Introduction, front matter)
- Venue (conference/journal/arXiv): "Proceedings of the 35<sup>th</sup> International Conference on Machine Learning, Stockholm, Sweden, PMLR 80, 2018." (Introduction, front matter)

---

## 2. One-Sentence Contribution Summary

The paper "generalize[s] a recently proposed model architecture based on self-attention, the Transformer, to a sequence modeling formulation of image generation with a tractable likelihood" and argues that "restricting the selfattention mechanism to attend to local neighborhoods" enables practical image-scale modeling improvements. (Abstract)

---

## 3. Tasks Evaluated

- Task name: Unconditional image generation / generative image modeling
  - Task type: Generation
  - Dataset(s) used: CIFAR-10; ImageNet
  - Domain: natural images
  - Quotes: "Our unconditioned and class-conditioned image generation models both use 1D local attention" (Section 5.1); "On CIFAR-10 our best unconditional models achieve a perplexity of 2.90 bits/dim on the test set" (Section 5.1); "On the more challenging ImageNet data set, however, the Image Transformer performs significantly better" (Section 5.1); "We trained only unconditional generative models on ImageNet" (Section 5.1)

- Task name: Class-conditional image generation
  - Task type: Generation
  - Dataset(s) used: CIFAR-10
  - Domain: natural images
  - Quotes: "In image-class conditional generation we condition on an embedding of one of a small number of image classes." (Introduction); "We trained the class-conditioned Image Transformer on CIFAR-10" (Section 5.2)

- Task name: Image completion (conditional generation of a missing region)
  - Task type: Generation; Other (image completion)
  - Dataset(s) used: CIFAR-10
  - Domain: natural images
  - Quotes: "Table 1. Three outputs of a CelebA super-resolution model followed by three image completions by a conditional CIFAR-10 model" (Table 1); "Table 2. On the left are image completions from our best conditional generation model, where we sample the second half." (Table 2)

- Task name: Image super-resolution (4x)
  - Task type: Generation; Reconstruction
  - Dataset(s) used: CelebA; CIFAR-10
  - Domain: natural images (faces for CelebA); natural images
  - Quotes: "We also present results on image super-resolution with a large magnification ratio (4x)" (Abstract); "Super-resolution is the process of recovering a high resolution image from a low resolution image" (Section 5.3); "We trained both our 1D Local and 2D Local models on the standard CelebA data set of celebrity faces" (Section 5.3); "CIFAR-10 We also trained a super-resolution model on the CIFAR-10 data set." (Section 5.3)

---

## 4. Domain and Modality Scope

- Single domain? No; evaluations span multiple image datasets: "On CIFAR-10" and "On the more challenging ImageNet data set" (Section 5.1) and "We trained both our 1D Local and 2D Local models on the standard CelebA data set of celebrity faces" (Section 5.3).
- Multiple domains within the same modality? Yes; CIFAR-10, ImageNet, and CelebA are all image datasets: "On CIFAR-10" (Section 5.1), "ImageNet" (Section 5.1), and "CelebA" (Section 5.3).
- Multiple modalities? No; tasks are image-only, e.g., "image generation" and "image super-resolution" (Abstract).
- Does the paper claim domain generalization or cross-domain transfer? Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Unconditional image generation | Not specified (separate training per dataset/task is described) | Not specified | Not specified | "On CIFAR-10 our best unconditional models achieve a perplexity of 2.90 bits/dim" and "We trained only unconditional generative models on ImageNet" (Section 5.1) |
| Class-conditional image generation | Not specified | Not specified | Not specified | "We trained the class-conditioned Image Transformer on CIFAR-10" (Section 5.2) |
| Image completion (conditional generation) | Not specified | Not specified | Not specified | "image completions by a conditional CIFAR-10 model" (Table 1); "image completions from our best conditional generation model" (Table 2) |
| Image super-resolution | Not specified (separate models per dataset/attention scheme are described) | Not specified | Not specified | "We trained both our 1D Local and 2D Local models on the standard CelebA data set" and "We also trained a super-resolution model on the CIFAR-10 data set" (Section 5.3) |

---

## 6. Input and Representation Constraints

- Fixed or variable input resolution? Variable in formulation but fixed in experiments: "For an image of width w and height h" (Section 3.1), and in experiments "we enlarge an  $8\times 8$  pixel image four-fold to  $32\times 32$" plus "we resized the image to  $8\times 8$  pixels for the input and  $32\times 32$  pixels for the label" (Section 5.3).
- Fixed patch size? Not specified; positions are per pixel/channel: "each channel of each pixel" (Section 3.2).
- Fixed number of tokens? Fixed by image size in experiments: "produce  $32\times 32$  pixel images with 3072 positions" and "operate on  $8\times 8$  pixel images ... 192 positions" (Section 3.3).
- Fixed dimensionality (e.g., strictly 2D)? Representations are 2D images with channel handling: "For an image of width w and height h, we combine the width and channel dimensions yielding a 3-dimensional tensor with shape  $[h, w \cdot 3, d]$" and "form an input representation with shape [h, w, d]" (Section 3.1).
- Any padding or resizing requirements? "padding with zeroes if necessary" (Section 3.3); "we resized the image to  $8\times 8$  pixels for the input and  $32\times 32$  pixels for the label" (Section 5.3).

---

## 7. Context Window and Attention Structure

- Maximum sequence length: "produce  $32\times 32$  pixel images with 3072 positions" (Section 3.3); encoder context for super-resolution is smaller: "operate on  $8\times 8$  pixel images ... 192 positions" (Section 3.3).
- Fixed or variable length? Variable with image size in formulation ("For an image of width w and height h") but fixed in experiments (" $32\times 32$  pixel images" and " $8\times 8$  pixel images") (Sections 3.1, 3.3).
- Attention type: Local/windowed self-attention in decoders ("restricting the positions in the memory matrix M to a local neighborhood around the query position" in Section 3.3; "1D local attention" and "2D local attention" in Section 3.3); global self-attention in encoder for super-resolution ("we don't require masking, but allow any input pixel to attend to any other pixel" in Section 5.3); encoder-decoder attention for conditional generation ("the decoder uses an attention mechanism to consume the encoder representation" in Section 3.2).
- Mechanisms to manage computational cost: local attention with block partitioning: "we partition the image into query blocks and associate each of these with a larger memory block" (Section 3.3); scalability tied to memory size: "The number of positions included in the memory  $l_m$ ... has tremendous impact on the scalability of the self-attention mechanism" (Section 3.3).

---

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: Absolute coordinate encodings, either sinusoidal or learned: "We evaluated two different coordinate encodings: sine and cosine functions of the coordinates ... and learned position embeddings." (Section 3.1)
- Where it is applied: Added to input representations; "To each pixel representation, we add a d-dimensional encoding of coordinates of that pixel." (Section 3.1) and "The position encodings  $p_q, p_1, \ldots$  are added only in the first layer." (Section 3.2)
- Fixed across all experiments or modified per task? Compared in some settings: "For experiments with the categorical distribution we evaluated both coordinate encoding schemes ... and found no difference in quality. For DMOL we only evaluated learned coordinate embeddings." (Section 5.1)

---

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption? Compared but not presented as a core variable: "We evaluated two different coordinate encodings: sine and cosine functions ... and learned position embeddings." (Section 3.1)
- Are multiple positional encodings compared? Yes: "For experiments with the categorical distribution we evaluated both coordinate encoding schemes ..." (Section 5.1)
- Claim PE choice is "not critical" or secondary? Implicitly secondary: "found no difference in quality" (Section 5.1)

---

## 10. Evidence of Constraint Masking

- Model sizes: "For categorical, we use 12 layers with d = 512, heads=4, feed-forward dimension 2048" and "In DMOL, our best config uses 14 layers, d = 256, heads=8" (Section 5.1); "Our ImageNet unconditioned generation model has 12 self-attention and feed-forward layers, d=512,8 attention heads, 2048 dimensions in the feed-forward layers" (Section 5.1).
- Dataset size (scale): "ImageNet is a much larger dataset, with many more categories than CIFAR-10, requiring more parameters in a generative model." (Section 5.1)
- Gains attributed to receptive field / memory size (architecture scaling): "Our experiments indicate that increasing the size of the receptive field plays a significant role in this improvement." (Introduction); "Table 4 shows that growing the receptive field improves perplexity significantly." (Section 5.1); "Increasing memory block size (bsize) significantly improves performance." (Table 4 caption)

---

## 11. Architectural Workarounds

- Local attention to scale to larger images: "By restricting the selfattention mechanism to attend to local neighborhoods we significantly increase the size of images the model can process in practice" (Abstract).
- Query/memory block partitioning for efficiency: "we partition the image into query blocks and associate each of these with a larger memory block" (Section 3.3).
- 1D and 2D local attention schemes: "In our experiments we use two different schemes... **1D Local Attention** ... **2D Local Attention**" (Section 3.3).
- Autoregressive masking: "we mask attention weights ... positions that have not yet been generated are ignored." (Section 3.3)
- Encoder-decoder for conditional generation (super-resolution): "For image-conditioned generation, as in our super-resolution models, we use an encoder-decoder architecture." (Section 3.2)
- Decoder-only for unconditional/class-conditional generation: "For unconditional and class-conditional generation, we employ the Image Transformer in a decoder-only configuration." (Section 3.2)
- Class-conditioning via embeddings: "We represent the image classes as learned d-dimensional embeddings per class and simply add the respective embedding to the input representation" (Section 5.2)

---

## 12. Explicit Limitations and Non-Claims

- Lack of ImageNet class labels for conditional training: "We trained only unconditional generative models on ImageNet, since class labels were not available in the dataset" (Section 5.1).
- Unimplemented architectural variants left for future work: "These modifications are readily applicable to our model, which we plan to evaluate in future work." (Section 2)
- Future work on text conditioning (not done in this paper): "In future work we would like to explore a broader variety of conditioning information including free-form text" (Conclusion).
- Future work beyond still images: "Fundamentally, we aim to move beyond still images to video" (Conclusion).

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: Multiple image datasets within one modality ("CIFAR-10," "ImageNet," "CelebA").
> – Task structure: Autoregressive image generation and conditional variants ("unconditioned and class-conditioned image generation"; "image super-resolution").
> – Representation rigidity: Fixed-size grids in experiments (" $8\times 8$ " to " $32\times 32$ "; "3072 positions") with pixel/channel tokens.
> – Model sharing vs specialization: Separate training described per task/dataset ("We trained the class-conditioned Image Transformer on CIFAR-10"; "We trained ... on the standard CelebA data set").
> – Role of positional encoding: Absolute coordinate encodings compared, with "no difference in quality" reported.

---

### 14. Final Classification

**Multi-task, multi-domain (constrained).** The paper evaluates multiple tasks including "unconditioned and class-conditioned image generation" and "image super-resolution" (Sections 5.1–5.3), and also presents "image completions" (Table 2). It spans multiple image domains/datasets ("CIFAR-10," "ImageNet," "CelebA") while staying within a single modality and without cross-domain transfer claims (Sections 5.1, 5.3).
