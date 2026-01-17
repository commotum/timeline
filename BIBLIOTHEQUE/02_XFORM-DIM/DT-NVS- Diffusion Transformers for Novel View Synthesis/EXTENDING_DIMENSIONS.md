## 1. Basic Metadata

- Title: "DT-NVS: Diffusion Transformers for Novel View Synthesis" (Title)
- Authors: "Wonbong Jang* Jonathan Tremblay† Lourdes Agapito*" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper proposes "DT-NVS, a novel view synthesis diffusion model that exploits a transformer-based backbone architecture to predict a radiance field from a single reference image" for "generating novel views of general scenes, from a single input image and using only 2D losses" (1 Introduction).

---

## 3. Tasks Evaluated

Task 1:
- Task name: "generalized novel view synthesis from a single input image" (Abstract).
- Task type: Generation; Reconstruction. Evidence: "generating novel views of general scenes, from a single input image and using only 2D losses." (1 Introduction); "predict a radiance field from a single reference image." (1 Introduction).
- Dataset(s) used: "MVImgNet consists of 6.5M images from 220K scenes across 238 cateogories, all of them are real world captures" (5.1 MVImgNet); "We use ShapeNet renderings from [2] to validate our model." (5.3 ShapeNet).
- Domain: "real world captures" (5.1 MVImgNet); "ShapeNet renderings" and "The dataset assumes an object-centric scene" (5.3 ShapeNet).

---

## 4. Domain and Modality Scope

- Evaluation domain scope: Multiple domains within the same modality (2D images), with "real world captures" in MVImgNet and "ShapeNet renderings" in ShapeNet (5.1 MVImgNet; 5.3 ShapeNet).
- Multiple modalities: Not specified beyond images; the paper frames inputs as "RGB renderings" (3 Background).
- Domain generalization or cross-domain transfer: The paper frames the task as "generalized novel view synthesis" (Abstract) and states the approach "can be applied to any real-world captures" (1 Introduction). Cross-domain transfer: Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Generalized novel view synthesis from a single input image | Not specified. | Not specified. | Not specified. | "We train our model using 2 A100-40GB GPUs for both MVImgNet and ShapeNet datasets, using the same architecture with both." (5.4 Implementation Details); "we train the model separately for each aspect ratio." (5.1 MVImgNet) |

---

## 6. Input and Representation Constraints

- Constant focal length assumption: "focal length f (which we assume constant)." (3 Background)
- Identity rotation and camera normalization: "we assume that input views  $c^i$  always have the identity rotation." (4.2 Predicting the scene: Transformer, Relative Pose, VM-Representation) and "we apply an affine transformation to move the input camera to be at  $(0,0,-r_d)$ , with identity rotation" (4.2 Predicting the scene: Transformer, Relative Pose, VM-Representation).
- Fixed preprocessing resolution/aspect ratio for MVImgNet: "The dataset contains both portrait and aspect ratio, and we train the model separately for each aspect ratio." and "We downsample and center-crop images to  $56 \times 32$  and  $32 \times 56$" (5.1 MVImgNet).
- Scene scaling/normalization: "we also downscale the point clouds from COLMAP to unit-cube, and change focal length accordingly." (5.1 MVImgNet)
- ShapeNet alignment and camera constraints: "The dataset assumes an object-centric scene" and "It also adopts the simplified camera model which always points toward the center of the coordinate system. Additionally, all 3D objects are aligned, sharing the same reference frame" (5.3 ShapeNet).
- Resolution limitation in practice: "We needed to downsample images significantly to train this model, resulting in a loss of output quality." (Limitation)
- Fixed patch size / fixed number of tokens: Not specified.

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Fixed or variable sequence length: Not specified.
- Attention type: "The decoder employs self-attention only, by concatenating feature tokens from the encoder with output tokens which are replicated for grid position, then differentiated by learnable positional embedding." (4.2 Predicting the scene: Transformer, Relative Pose, VM-Representation)
- Computational cost management: "Representing the scene as a voxel-grid is computationally expensive. ... We adopt the Vector-Matrix Representation (VM Representation) proposed by TensoRF [9]." (4.2 Predicting the scene: Transformer, Relative Pose, VM-Representation)

---

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: "learnable positional embedding." (4.2 Predicting the scene: Transformer, Relative Pose, VM-Representation); type (absolute/relative/etc.) not specified.
- Where it is applied: "output tokens which are replicated for grid position, then differentiated by learnable positional embedding." (4.2 Predicting the scene: Transformer, Relative Pose, VM-Representation)
- Fixed across all experiments / modified per task / ablated: Not specified.

---

## 9. Positional Encoding as a Variable

- Core research variable? Not stated; the only explicit mention is "learnable positional embedding." (4.2 Predicting the scene: Transformer, Relative Pose, VM-Representation)
- Fixed architectural assumption? Not stated.
- Multiple positional encodings compared? Not stated.
- PE choice claimed "not critical" or secondary? Not stated.

---

## 10. Evidence of Constraint Masking

- Model size(s): Not specified.
- Dataset size(s): "MVImgNet consists of 6.5M images from 220K scenes across 238 cateogories" (5.1 MVImgNet); "The dataset includes three categories (cars, chairs and planes) with each category containing 3,200 scenes, divided into training (2700) and testing (500) sets." (5.3 ShapeNet).
- Training scale: "We train our model using 2 A100-40GB GPUs for both MVImgNet and ShapeNet datasets, using the same architecture with both. Training MVImgNet takes 5 days (700,000 iterations) for both landscape and portrait, while training ShapeNet takes 2 days (400,000 iterations). We use a batch size of 44 for MVImgNet and 26 for ShapeNet." (5.4 Implementation Details)
- Attribution of gains: The paper emphasizes design/training choices, e.g., "Randomly swapping the reference frame between the reference image and the sampled noisy input image regularizes the model for better performance" (5.2 Ablation Study) and "Without the encoder, the model's performance significantly drops." (5.2 Ablation Study). Scaling model size or data is not claimed as the primary driver.

---

## 11. Architectural Workarounds

- Relative pose normalization: "we assume that input views  $c^i$  always have the identity rotation" and "we apply an affine transformation to move the input camera to be at  $(0,0,-r_d)$ , with identity rotation" (4.2 Predicting the scene: Transformer, Relative Pose, VM-Representation).
- Camera-parameter conditioning: "The decoder conditions on camera parameters using adaptive layer normalization (AdaLN) [73]." (4.2 Predicting the scene: Transformer, Relative Pose, VM-Representation)
- Self-attention-only decoder: "The decoder employs self-attention only" (4.2 Predicting the scene: Transformer, Relative Pose, VM-Representation).
- Swapping input/reference and dropout regularization: "We randomly swap the positions of noisy input images with reference images during the training" and "we apply dropout to reference images, which regularizes the model and enables it to perform unconditional generation and classifier-free guidance." (4.2 Predicting the scene: Transformer, Relative Pose, VM-Representation)
- VM representation for efficiency: "Representing the scene as a voxel-grid is computationally expensive. ... We adopt the Vector-Matrix Representation (VM Representation) proposed by TensoRF [9]." (4.2 Predicting the scene: Transformer, Relative Pose, VM-Representation)

---

## 12. Explicit Limitations and Non-Claims

- "We needed to downsample images significantly to train this model, resulting in a loss of output quality." (Limitation)
- "The model also occasionally struggles with outdoor scenes, partly due to the VM representation." (Limitation)
- "Recent diffusion approaches, like flow models [35], could not be employed." (Limitation)
- Future work: "An interesting future direction includes extending this model to more challenging outdoor scenes, higher resolution or associating with other modalities." (6 Conclusion)
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not stated.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Evaluated on "real world captures" (MVImgNet) and "ShapeNet renderings" (5.1, 5.3), both image modality.
> - Task structure: Single task of "generalized novel view synthesis from a single input image" (Abstract).
> - Representation rigidity: Assumes constant focal length and identity-rotation input views, plus fixed low-resolution preprocessing and object-centric alignment in ShapeNet (3 Background; 4.2; 5.1; 5.3).
> - Model sharing vs specialization: Same architecture across datasets but trained separately per aspect ratio; weight sharing across tasks is not specified (5.1; 5.4).
> - Role of positional encoding: Only a "learnable positional embedding" is mentioned; no variation or comparison is described (4.2).

---

### 14. Final Classification

**Classification:** Single-task, single-domain

Justification: The paper evaluates a single task, "generalized novel view synthesis from a single input image" (Abstract). Evaluation uses MVImgNet "real world captures" and ShapeNet "renderings" (5.1; 5.3), but both are image-based novel view synthesis and the paper does not claim multi-task learning or cross-domain transfer.
