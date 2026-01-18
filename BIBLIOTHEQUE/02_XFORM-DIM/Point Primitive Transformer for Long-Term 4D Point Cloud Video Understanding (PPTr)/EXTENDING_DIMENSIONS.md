## 1. Basic Metadata

- Title: Point Primitive Transformer for Long-Term 4D Point Cloud Video Understanding
- Authors: Hao Wen, Yunze Liu, Jingwei Huang, Bo Duan, and Li Yi
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

"This paper proposes a 4D backbone for long-term point cloud video understanding." (Abstract)

## 3. Tasks Evaluated

### Task: 4D Semantic Segmentation
- Task name: 4D semantic segmentation
- Task type: Segmentation
- Dataset(s) used: Synthia4D; HOI4D
- Domain: 4D point cloud video (synthetic driving scenes; egocentric human-object interaction)
- Evidence:
  - "4D semantic segmentation on Synthia4D [34] and HOI4D [29]." (Section 1 Introduction)
  - "Synthia 4D [34] is a synthetic dataset for outdoor autonomous driving." (Section 5.1)
  - "HOI4D dataset for experiments, which is a large-scale 4D egocentric dataset to catalyze the research of category-level human-object interaction. It provides frame-wise annotations for 4D point cloud semantic segmentation." (Section 5.1)

### Task: 3D Action Recognition
- Task name: 3D action recognition
- Task type: Classification
- Dataset(s) used: MSR-Action [25]; MAR-Action3D
- Domain: human body point cloud videos
- Evidence:
  - "3D action recognition on MSR-Action [25]." (Section 1 Introduction)
  - "we use the MAR-Action3D dataset which consists of 567 human body point cloud videos, including 20 action categories." (Section 5.2)
  - "We use the video classification accuracy as the evaluation metric." (Section 5.2)

## 4. Domain and Modality Scope

- Evaluation is performed on: Multiple domains within the same modality (point cloud video). Evidence: "3D action recognition on MSR-Action [25] and 4D semantic segmentation on Synthia4D [34] and HOI4D [29]." (Section 1 Introduction)
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| 4D semantic segmentation | Not specified. | Not specified. | Yes. | "PPTr is very flexible for both point-wise and sequence-wise inference by simply changing the task head." (Section 4 Method) "For semantic segmentation, primitive features are concatenated to corresponding point features then classified into semantic labels." (Fig. 4 caption) |
| 3D action recognition | Not specified. | Not specified. | Yes. | "PPTr is very flexible for both point-wise and sequence-wise inference by simply changing the task head." (Section 4 Method) "For action recognition, primitive features are merged by maxpooling to a global feature then classified into actions." (Fig. 4 caption) |

## 6. Input and Representation Constraints

- Short clip input: "The input to network is a short video clip." (Fig. 4 caption) and "PPTr extracts short-term spatial-temporal features through an intra-primitive point transformer for a short video clip around the frame of interest." (Fig. 1 caption)
- Sequence representation and variable length: "We represent a point cloud sequence as  $\Psi = \{(P_t, V_t) | t = 1, ..., L\}$ ." (Section 4.1)
- Primitive-plane representation with explicit labels/parameters: "we detect planes for each frame  $(P_t, V_t)$  and output primitive label  $\Xi_t \in \mathbb{R}^{N \times 3}$  and primitive parameters  $\Theta_t \in \mathbb{R}^{M \times 4}$ , where N is the number of points and M is the number of primitives." (Section 4.1)
- Fixed primitives per task: "In this task, we fit the scene point cloud into 200 primitives." (Section 5.1) and "We fit the human body point cloud into 4 primitives." (Section 5.2)
- Fixed points per frame (action recognition): "Each frame is sampled with 2,048 points." (Section 5.2)
- Clip segmentation: "As inputs, point cloud videos are split into multiple clips." (Section 5.2)
- Primitive memory pool size depends on L and M: "The final memory pool  $F_{\text{mem}}$  has a shape of  $\mathbb{R}^{C \times M \times L}$ ." (Section 4.1)

## 7. Context Window and Attention Structure

- Maximum sequence length: 30 frames in experiments. Evidence: "When using the memory pool to integrate temporal information from 30 frames" and "our method is the first to integrate point clouds of 30 frames." (Section 5.1)
- Sequence length fixed or variable: Variable (denoted by L / L'). Evidence: "$\Psi = \{(P_t, V_t) | t = 1, ..., L\}$" (Section 4.1) and "$F_{\text{in}}^{\text{primitive}} = [F_{\text{clip}}||F_{\text{mem}}] \in \mathbb{R}^{C^l \times (L'+L)M}$" (Section 4.3)
- Attention type: Hierarchical with local (intra-primitive) and cross-primitive self-attention. Evidence: "Point Primitive Transformer(PPTr) is a two-level hierarchical transformer" and "The intra-primitive point transformer restricts the communication of points within each primitive plane." (Section 4 Method) plus "clip primitive embeddings(green) perform self-attention with long-term embeddings(yellow) in the memory pool." (Fig. 4 caption)
- Computational cost mechanisms: locality + memory pool + offline branch. Evidence: "Pre-computed primitive features allow aggregating long-term spatial-temporal context efficiently and effectively." (Section 4 Method) and "The backbone consists of two branches: online network and offline pre-computation." (Fig. 4 caption)

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: Not specified.
- Where it is applied: Not specified.
- Fixed across experiments / modified per task / ablated: Not specified.

## 9. Positional Encoding as a Variable

- Positional encoding as a core research variable or fixed assumption: Not specified.
- Multiple positional encodings compared: Not specified.
- Claims PE choice is not critical or secondary: Not specified.

## 10. Evidence of Constraint Masking

- Model size(s): Not specified.
- Dataset size(s): "with 19,888/815/1,886 frames, respectively" for Synthia 4D (Section 5.1); "request 1000 sequences, which includes 30k frames of the point cloud" for HOI4D (Section 5.1); "567 human body point cloud videos, including 20 action categories" for MAR-Action3D (Section 5.2).
- Performance gains attributed to architecture and temporal scaling: "Our PPTr with 1 frame can achieve 0.69% improvement over the P4Transformer with 3 frames, which demonstrates the effectiveness of the hierarchical structure." (Section 5.1) and "When using the memory pool to integrate temporal information from 30 frames, we can achieve 1.33% improvement" (Section 5.1).
- Efficiency constraints and workaround: "For the 4D segmentation task, using our online branch independently, the memory could only afford 3 frames" and "assisted by the offline branch covering 30 frames" (Section 5.3).

## 11. Architectural Workarounds

- Hierarchical two-level transformer (point + primitive): "Point Primitive Transformer(PPTr) is a two-level hierarchical transformer" (Section 4 Method).
- Intra-primitive locality to restrict attention: "The intra-primitive point transformer restricts the communication of points within each primitive plane." (Section 4 Method)
- Memory pool for long-term context: "Pre-computed primitive features allow aggregating long-term spatial-temporal context efficiently and effectively." (Section 4 Method)
- Online/offline branches to manage cost: "The backbone consists of two branches: online network and offline pre-computation." (Fig. 4 caption)
- Task-specific heads: "For semantic segmentation, primitive features are concatenated to corresponding point features then classified into semantic labels. For action recognition, primitive features are merged by maxpooling to a global feature then classified into actions." (Fig. 4 caption)

## 12. Explicit Limitations and Non-Claims

- Limitation (online branch memory): "For the 4D segmentation task, using our online branch independently, the memory could only afford 3 frames" (Section 5.3).
- Future work: "suggests future work to explore more possible backbone designs for 4D point cloud understanding." (Section 6 Conclusions)
- Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Multiple domains within a single modality (point cloud video) across Synthia4D, HOI4D, and MSR-Action [25] and MAR-Action3D.
> - Two tasks (4D semantic segmentation and 3D action recognition) with task-specific heads rather than a unified multi-task objective.
> - Representation is structured around primitive planes, with fixed primitive counts per task (e.g., 200 for scenes; 4 for human bodies) and fixed 2,048 points per frame in action recognition.
> - Backbone is shared conceptually but weights/joint training across tasks are not specified; task heads differ.
> - Positional encoding is not discussed, implying it is not treated as a variable in the reported experiments.

### 14. Final Classification

**Multi-task, multi-domain (constrained).** The paper evaluates both "3D action recognition" and "4D semantic segmentation" across different datasets/domains (MSR-Action [25] and MAR-Action3D, Synthia4D, HOI4D) while staying within point cloud video data (Section 1 Introduction; Section 5.1; Section 5.2). The setup remains constrained to a single modality with task-specific heads rather than unrestrained multi-task learning.
