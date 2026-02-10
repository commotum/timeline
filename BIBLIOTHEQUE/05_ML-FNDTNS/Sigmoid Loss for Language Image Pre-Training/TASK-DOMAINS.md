# Sigmoid Loss for Language Image Pre-Training (2023)
Source: Sigmoid Loss for Language Image Pre-Training.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Contrastive language-image pair matching (pre-training) | Images and tokenized text pairs | 2D (x, y); 1D (t) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Pair-match scores/labels | 0D | Capped (inferred) |
| Zero-shot image classification | Images | 2D (x, y) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Class labels | 0D | Capped (inferred) |
| Zero-shot image-to-text retrieval | Images and candidate texts | 2D (x, y); 1D (t) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Ranked text items | 1D (t) | Capped (inferred) |
| Zero-shot text-to-image retrieval | Text queries and candidate images | 1D (t); 2D (x, y) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Ranked image items | 2D (x, y) | Capped (inferred) |

## Summary
The paper covers multimodal language-image contrastive pre-training and evaluates downstream zero-shot transfer on classification and bidirectional retrieval. Inputs span images (2D (x, y)) and tokenized text (1D (t)), with outputs spanning scalar labels/scores (0D) and ranked cross-modal retrieval items (1D text or 2D images). Interface dynamics are mostly Capped (and Fixed for resized image-only classification input), based on explicit token limits, resized image inputs, and finite benchmark candidate sets. Attention is Static and state is Constructed in all rows as an inference from fixed-encoder processing and explicit learned feature-vector representations.

## Evidence
### Task: Contrastive language-image pair matching (pre-training)
- "Given a mini-batch  $\mathcal{B}=\{(I_1,T_1),(I_2,T_2),\dots\}$  of image-text pairs, the contrastive learning objective encourages embeddings of matching pairs  $(I_i,T_i)$  to align with each other, while pushing embeddings of unmatched pairs  $(I_i,T_{j\neq i})$  apart." (Section 3)
- "The sigmoid-based loss processes every image-text pair independently, effectively turning the learning problem into the standard binary classification on the dataset of all pair combinations, with a positive labels for the matching pairs  $(I_i, T_i)$  and negative labels for all other pairs  $(I_i, T_{i \neq i})$ ." (Section 3.2)
- "where  $z_{ij}$  is the label for a given image and text input, which equals 1 if they are paired and -1 otherwise." (Section 3.2)
- Inference: In Dynamics is marked Capped because the text side is explicitly bounded ("a maximum of 16 text tokens are kept"), while Attention Dynamic is Static and State Dynamic is Constructed from the fixed-encoder setup and learned embedding-space operation.

### Task: Zero-shot image classification
- "To validate our models, we report zero-shot transfer results on the ImageNet dataset [14] and zero-shot retrieval results across 36 languages on the XM3600 dataset [44]." (Section 4)
- "The models can be used for zero-shot image classification and zero-shot imagetext retrieval by comparing both feature vectors." (Section F. Model Card)
- "The vision encoder takes an image (224 × 224 × 3, 256 × 256 × 3, 384 × 384 × 3, 512 × 512 × 3) as input." (Section F. Model Card)
- Inference: In Dynamics is Fixed from resized fixed-resolution image inputs; Attention Dynamic is Static because no runtime input-selection mechanism is described; State Dynamic is Constructed due to learned feature vectors; Out Dynamics is Capped because classification is over finite benchmark label sets.

### Task: Zero-shot image-to-text retrieval
- "In Table 3, we report zero-shot classification results on ImageNet [14], ObjectNet [2], ImageNet-v2 [39], ImageNet ReaL [3], and zero-shot image-to-text ( $I \rightarrow T$ ) retrieval, texto-image ( $I \rightarrow T$ ) retrieval results on MSCOCO [11]." (Section 4.6)
- "Figure 8: Image-to-text and text-to-image zero-shot retrieval recall@1 results on all 36 languages of Crossmodal-3600. Top: Image to text." (Appendix, Figure 8)
- "- Intended Use: The models are designed for multimodal research purposes. The models can be used for zero-shot image classification and zero-shot imagetext retrieval by comparing both feature vectors." (Section F. Model Card)
- Inference: Dynamics are marked Capped from finite retrieval candidate pools in reported benchmarks and recall@1 reporting; Attention Dynamic is Static and State Dynamic is Constructed from fixed embedding computation and feature-comparison retrieval.

### Task: Zero-shot text-to-image retrieval
- "Figure 8: Image-to-text and text-to-image zero-shot retrieval recall@1 results on all 36 languages of Crossmodal-3600. Top: Image to text. Bottom: text to image." (Appendix, Figure 8)
- "Table 9: Image-to-text (text retrieval) and text-to-image (image retrieval) zero-shot recall@1 results on all 36 languages of Crossmodal-3600, with mSigLIP models trained at different batch sizes for 30 B total examples seen." (Appendix, Table 9)
- "We also scale up the multilingual mSigLIP ViT-B model in the same way. We report image-text retrieval results across 36 languages on the XM3600 benchmark [44]." (Section 4.6)
- Inference: Dynamics are Capped from finite candidate sets and recall@1 evaluation; Attention Dynamic is Static and State Dynamic is Constructed due to fixed dual-encoder feature construction followed by similarity-based ranking.
