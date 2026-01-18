## 1. Basic Metadata

- Title: "Positional Encoding Field" (Title block)
- Authors: "Yunpeng Bai", "Haoxiang Li", "Qixing Huang" (Title block: "Yunpeng Bai* University of Texas at Austin", "Haoxiang Li Pixocial Technology", "Qixing Huang University of Texas at Austin")
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper introduces the "Positional Encoding Field (PE-Field), which extends positional encodings from the 2D plane to a structured 3D field," and reports that the resulting DiT "achieves state-of-the-art performance on single-image novel view synthesis and generalizes to controllable spatial image editing" (Abstract).

## 3. Tasks Evaluated

Task name: Single-image novel view synthesis (NVS).
Task type: Generation; Reconstruction.
Dataset(s) used: DL3DV, MannequinChallenge (training); Tanks-and-Temples, RE10K, DL3DV (evaluation).
Domain: images (single input image, novel viewpoints).
Quotes: "In this work, we mainly want to leverage these findings to address novel view synthesis (NVS) problem from a single image." (Section 3.1) "To train our NVS model, we use two multi-view datasets, DL3DV [19] and MannequinChallenge [17]" (Section 4.1) "Experiments are conducted on three datasets, Tanks-and-Temples [14], RE10K [54], and DL3DV [19]. In each case, a single input image is provided, and subsequent frames are generated under different target viewpoints." (Section 4.2)

Task name: Object-level 3D editing.
Task type: Generation; Other (specify: spatial image editing).
Dataset(s) used: Not specified.
Domain: 3D point cloud with image background.
Quotes: "we perform object-level 3D editing by isolating the point cloud of the book, rotating it to a new viewpoint, and recomposing it with the original background." (Section 4.4)

Task name: Object removal.
Task type: Generation; Other (specify: spatial image editing).
Dataset(s) used: Not specified.
Domain: image tokens with masked regions.
Quotes: "we achieve object removal by discarding the tokens corresponding to the masked human region and replenishing them with noise, resulting in a realistic removal effect." (Section 4.4)

## 4. Domain and Modality Scope

- Evaluation is on multiple datasets within the same modality (images), not multiple modalities: "Experiments are conducted on three datasets, Tanks-and-Temples [14], RE10K [54], and DL3DV [19]. In each case, a single input image is provided" (Section 4.2).
- Depth is derived from images, but no multi-modality evaluation is claimed: "both processed with VGGT [40] to obtain perimage depth maps and corresponding camera poses." (Section 4.1)
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Single-image novel view synthesis (NVS) | Yes (same NVS model reused for other tasks) | Not specified | Not specified | "After training, our NVS model acquires the ability to reason over visual tokens in 3D space and generate consistent content. Consequently, it can naturally adapt to other tasks with similar spatial logic, even in the absence of task-specific training." (Section 4.4) "To train our NVS model, we use two multi-view datasets, DL3DV [19] and MannequinChallenge [17]" (Section 4.1) |
| Object-level 3D editing | Yes | No (absence of task-specific training) | Not specified | "Consequently, it can naturally adapt to other tasks with similar spatial logic, even in the absence of task-specific training." (Section 4.4) "we perform object-level 3D editing by isolating the point cloud of the book, rotating it to a new viewpoint, and recomposing it with the original background." (Section 4.4) |
| Object removal | Yes | No (absence of task-specific training) | Not specified | "Consequently, it can naturally adapt to other tasks with similar spatial logic, even in the absence of task-specific training." (Section 4.4) "we achieve object removal by discarding the tokens corresponding to the masked human region and replenishing them with noise, resulting in a realistic removal effect." (Section 4.4) |

## 6. Input and Representation Constraints

- Patch-tokenized image representation: "By encoding images into sequences of patch tokens and applying 2D positional encodings (PEs) [38]" (Introduction); "each image patch is represented as a single token, i.e., a one-dimensional vector  $\mathbf{x}_i \in \mathbb{R}^d$" (Section 3.2).
- Patch size and sub-patch granularity are tied to PE levels: "positional grids from patch tokens (e.g.,  $16 \times 16$  pixels) are coarser than dense 3D reconstructions" (Section 3.1); "The coarsest level corresponds to a  $16\times 16$ -pixel patch, while the finest level corresponds to a  $4\times 4$ -pixel patch." (Section 3.2)
- Fixed 2D grid placement with discard/fill rules: "noise tokens are placed on a regular 2D grid with depth initialized to zero" and "Tokens projected outside the valid grid are discarded, and empty positions are filled with noise tokens" (Section 3.4).
- 3D positional coordinates with depth: "Each image token is assigned a hierarchical 3D positional encoding (x, y, z)that captures its detailed target spatial location and depth." (Section 3.4)
- Fixed or variable input resolution, fixed number of tokens, padding/resizing: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Fixed or variable sequence length: Not specified (sequence length shown as T in "X \in  $\mathbb{R}^{B \times T \times d}$" (Section 3.2)).
- Attention type: Multi-head self-attention is described, with no windowed/sparse/hierarchical attention specified: "Within the transformer, multi-head self-attention (MHA) is applied" (Section 3.2).
- Computational cost mechanisms (windowing, pooling, pruning): Not specified.

## 8. Positional Encoding (Critical Section)

- Mechanism: RoPE with hierarchical and depth-aware extensions: "we extend standard 2D RoPE [35] to a 3D depth-aware encoding" (Introduction); "augmenting it with multi-level hierarchical positional encodings" (Section 3.2); "we extend RoPE to include a third spatial axis for depth" (Section 3.3).
- Where applied: RoPE is applied to attention queries and keys per head: "Queries and keys in head h are rotated by the level-specific RoPE" (Section 3.2); "Each coordinate (x, y, z) thus has its own 1D RoPE encoding" (Section 3.3).
- Fixed vs modified across experiments: The PE choice is ablated and compared: "We mainly analyze the effect of removing our two key components: the hierarchical detailed positional encodings and the additional depth-aware extension." (Section 4.3) Table 1 lists "Original PE", "w/o Depth", "w/o Multi-Level", and "Ours" (Table 1).

## 9. Positional Encoding as a Variable

- Core research variable: Yes. "we introduce the Positional Encoding Field (PE-Field), which extends positional encodings from the 2D plane to a structured 3D field." (Abstract)
- Multiple positional encodings compared: Yes. "We mainly analyze the effect of removing our two key components: the hierarchical detailed positional encodings and the additional depth-aware extension." (Section 4.3) Table 1 lists "Original PE", "w/o Depth", "w/o Multi-Level", and "Ours" (Table 1).
- Claims that PE choice is not critical or secondary: Not stated.

## 10. Evidence of Constraint Masking

- Model size(s): Not specified.
- Dataset size(s): Not specified.
- Performance gains attributed to architecture components (PE-Field) rather than scaling: "We mainly analyze the effect of removing our two key components: the hierarchical detailed positional encodings and the additional depth-aware extension." (Section 4.3) "when the multi-level positional encoding (particularly the detailed level) is removed, undesirable distortions appear" (Section 4.3) and "When depth information is removed ... the generated images suffer from severe spatial misalignment." (Section 4.3)
- Claims about scaling model size or data: Not specified.

## 11. Architectural Workarounds

- Hierarchical multi-level RoPE for sub-patch detail: "augmenting it with multi-level hierarchical positional encodings" and "The coarsest level corresponds to a  $16\times 16$ -pixel patch, while the finest level corresponds to a  $4\times 4$ -pixel patch." (Section 3.2)
- Depth-aware 3D RoPE for volumetric reasoning: "we extend RoPE to include a third spatial axis for depth" and "This extension yields a 3D spatial RoPE that encodes relative offsets not only in the image plane but also along the depth axis" (Section 3.3).
- Token position reassignment for view synthesis: "we reassign positional encodings so that tokens migrate to their new projected locations." (Section 3.1)
- Fixed grid token handling to integrate observed and generated content: "noise tokens are placed on a regular 2D grid with depth initialized to zero" and "Tokens projected outside the valid grid are discarded, and empty positions are filled with noise tokens" (Section 3.4).
- Multi-step generation for large viewpoint changes: "we divide the transformation of the target viewpoint into five steps. After each step, the newly generated content is fused back into the image tokens of the original viewpoint" (Section 4.3).

## 12. Explicit Limitations and Non-Claims

- Limitation from patch-level manipulation and depth ambiguity: "artifacts remain due to: (1) resolution mismatch—positional grids from patch tokens (e.g.,  $16 \times 16$  pixels) are coarser than dense 3D reconstructions, limiting alignment precision. The manipulation can only rearrange image content at the patch level, but it cannot alter the content within each patch. and (2) depth ambiguity—multiple 3D points may project to the same token location. Without explicit mechanisms to disambiguate depth, generated tokens can collapse into inconsistent local structures." (Section 3.1)
- Limitation for large viewpoint changes: "When applying our method to generate results under large viewpoint changes, the model is required to directly generate a substantial amount of unseen content, which increases the generation burden and may compromise consistency with the source image." (Section 4.3)
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Single-modality images across multiple datasets; evaluation uses single input images for NVS.
> - Task structure: Primary single-image NVS with additional spatial editing tasks (object-level 3D editing, object removal) shown post-training.
> - Representation rigidity: Patch-token grid with fixed patch sizes and hierarchical 3D RoPE (x, y, z); tokens outside the grid are discarded and empty positions filled.
> - Model sharing vs specialization: One NVS model reused for editing tasks without task-specific training; no separate heads specified.
> - Role of positional encoding: Central variable (PE-Field) with explicit ablations on depth-aware and multi-level encodings.

### 14. Final Classification

Final classification: **Multi-task, single-domain**. The paper evaluates single-image novel view synthesis and additional spatial editing tasks, including "object-level 3D editing" and "object removal," using the same visual input modality where "a single input image is provided" (Section 4.2; Section 4.4). Evaluation spans multiple image datasets ("Tanks-and-Temples [14], RE10K [54], and DL3DV [19]") and does not claim cross-domain transfer (Section 4.2).
