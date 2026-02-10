# Value-Based Pre-Training with Downstream Feedback (Year not specified in the paper.)
Source: Value-Based Pre-Training with Downstream Feedback.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language modeling continued pretraining (next-token prediction with adaptive soft targets) | token sequences from an unlabeled math corpus | 1D (t) | Fixed | Static (inferred) | Direct (inferred) | next-token target distributions / token predictions | 1D (t) | Fixed |
| Mathematical reasoning answer generation (GSM8K, OMEGA) | math question prompts (text tokens) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | generated final answers (text / numeric tokens) | 1D (t) (inferred) | Not specified in the paper. |
| Multiple-choice question answering (MMLU) | question and answer-option text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | selected answer option | 0D (inferred) | Fixed (inferred) |
| Vision self-supervised representation learning (DINO-style SSL with learned views) | images with learned crops/masks | 2D (x, y) (inferred) | Fixed | Dynamic (inferred) | Direct (inferred) | self-supervised feature/projection targets | 1D (t) (inferred) | Fixed |
| Semantic segmentation (ADE20K) | images | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | semantic segmentation maps | 2D (x, y) (inferred) | Not specified in the paper. |
| Depth estimation (NYUv2) | images | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | depth maps / depth values | 2D (x, y) (inferred) | Not specified in the paper. |
| Image classification (ImageNet-1K linear evaluation) | images | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | class label | 0D (inferred) | Fixed (inferred) |
| Instance retrieval (Revisited Oxford/Paris) | query and database images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | ranked image list | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper covers both language and vision domains: controlled language-model pretraining and reasoning evaluation, plus vision SSL pretraining and dense/general-purpose vision transfer tasks. The justified input/output address spaces span 1D token sequences, 2D images, 2D dense maps, 0D class/choice outputs, and 1D ranked retrieval outputs. Dynamics are explicitly Fixed for compute-matched pretraining interfaces and Capped where context/database bounds are described or implied, while several downstream evaluation interfaces are not fully specified. Attention is mostly Static for fixed prompt/image processing, with Dynamic attention behavior inferred for vision learned-view generation that adapts per instance.

## Evidence
### Task: Language modeling continued pretraining (next-token prediction with adaptive soft targets)
- "Language: next-token prediction. Let  $x = (w_1, \ldots, w_T)$  be a sequence of discrete tokens." (Section 2.1)
- "In language, the designer replaces onehot next-token labels with adaptive soft targets supported on the learner's top-K candidates." (Section 1, Introduction)
- "are formatted as \"Question: ...\\n Answer: ...\" and packed to fixed length." (Section 4.1)
- Inference: `Attention Dynamic = Static` and `State Dynamic = Direct` are inferred from the fixed next-token predictive setup (causal context to token target) without runtime retrieval/tool selection. (Sections 2.1, 3.3)

### Task: Mathematical reasoning answer generation (GSM8K, OMEGA)
- "V-Pretraining of 0.5B-7B language models improves reasoning (GSM8K test Pass@1) by up to 18% relative over standard next-token prediction using only 12% of GSM8K training examples as feedback." (Abstract)
- "Evaluation uses GSM8K test Pass@1 with greedy decoding." (Section 4.1)
- "We use a fixed prompt that requests a single final answer, decode with greedy generation, and score exact match after normalization." (Section 4.3)
- "We tokenize with left padding and truncate the input to fit the model context limit." (Appendix A.4)
- Inference: `In Dimension = 1D (t)` and `Out Dimension = 1D (t)` come from text prompt/answer sequences; `In Dynamics = Capped` is inferred from explicit context-limit truncation. (Section 4.3; Appendix A.4)

### Task: Multiple-choice question answering (MMLU)
- "*Table 2.* **Evaluation on tasks not used for feedback.** Language: value-adjacent transfer under distribution shift (OMEGA) and value-extrapolative evaluation (MMLU)." (Table 2 caption, Section 4.2)
- "For value extrapolative transfer, we evaluate on MMLU using a standard zero shot multiple choice protocol." (Section 4.3)
- Inference: `Input` is question/options text, `Out Dimension = 0D`, and `Out Dynamics = Fixed` are inferred from the explicitly stated multiple-choice protocol (single selected option). (Section 4.3)

### Task: Vision self-supervised representation learning (DINO-style SSL with learned views)
- "Vision. Our baseline starts from DINOv3 pretrained ViT backbones (Siméoni et al., 2025) and continue SSL on ImageNet1K (Deng et al., 2009) using a DINO-style objective (Caron et al., 2021)." (Section 4.1)
- "Given an image x, the task designer outputs instance-specific augmentations that generate correlated views used by a standard SSL objective." (Section 3.3)
- "The baseline uses the standard fixed multi-crop augmentation pipeline, and V-Pretraining replaces fixed view generation with an instance-wise learned masking module." (Appendix A.2)
- Inference: `In Dimension = 2D (x, y)` is inferred from image input; `Attention Dynamic = Dynamic` is inferred because view/mask generation is instance-specific at runtime; `Out Dimension = 1D (t)` is inferred from fixed-dimensional projection/feature targets. (Section 3.3; Appendix A.2)

### Task: Semantic segmentation (ADE20K)
- "Downstream feedback uses small labeled pools from ADE20K segmentation (Zhou et al., 2017) and NYUv2 depth (Nathan Silberman & Fergus, 2012) to compute  $g_{\text{down}}$." (Section 4.1)
- "Using only 512 ADE20K and 512 NYUv2 images for feedback, value-based task design improves both ADE20K segmentation and NYUv2 depth relative to fixed augmentation baselines." (Section 4.2)
- "We evaluate representation quality using: (i) ADE20K mIoU with standard label remapping (ignore void) and either a linear-BN probe or a small conv decoder" (Appendix A.2)
- Inference: `In Dimension = 2D (x, y)` and `Out Dimension = 2D (x, y)` are inferred from image-to-segmentation mapping. (Section 4.2; Appendix A.2)

### Task: Depth estimation (NYUv2)
- "Downstream evaluators (dense tasks). We use two dense evaluators to define the value signal: ADE20K semantic segmentation and NYUv2 depth prediction." (Appendix A.2)
- "Vision: ADE20K mIoU, NYUv2 RMSE, and ImageNet linear accuracy." (Table 1 caption, Section 4.2)
- "(ii) NYUv2 depth using RMSE (and auxiliary metrics such as AbsRel and  $\delta_1$ )" (Appendix A.2)
- Inference: `In Dimension = 2D (x, y)` and `Out Dimension = 2D (x, y)` are inferred from image-to-depth-map prediction. (Appendix A.2)

### Task: Image classification (ImageNet-1K linear evaluation)
- "In vision SSL, we improve the state-of-the-art results on ADE20K by up to 1.07 mIoU and reduce NYUv2 RMSE while improving ImageNet linear accuracy" (Abstract)
- "We report ADE20K mIoU, NYUv2 RMSE, ImageNet linear accuracy, and instance retrieval transfer." (Section 4.1)
- "(iii) ImageNet-1K linear evaluation with a linear-BN head trained on frozen features" (Appendix A.2)
- Inference: `Out Dimension = 0D` and `Out Dynamics = Fixed` are inferred from single-label linear classification evaluation. (Appendix A.2)

### Task: Instance retrieval (Revisited Oxford/Paris)
- "*Table 2.* **Evaluation on tasks not used for feedback.** Language: value-adjacent transfer under distribution shift (OMEGA) and value-extrapolative evaluation (MMLU). Vision: instance retrieval transfer on Revisited Oxford/Paris." (Table 2 caption, Section 4.2)
- "evaluate frozen ViT-L representations on Revisited Oxford (R-Oxford5k) and Revisited Paris (R-Paris6k) instance retrieval" (Section 4.3)
- "rank database images by cosine similarity." (Section 4.3)
- Inference: `Out Dimension = 1D (t)` is inferred from ranked-list output; `In/Out Dynamics = Capped` is inferred from retrieval over finite benchmark databases. (Section 4.3)
