## 1. Basic Metadata

- Title: "Flamingo: a Visual Language Model for Few-Shot Learning" (Title header)
- Authors: "Jean-Baptiste Alayrac\*,‡ Jeff Donahue\* Antoine Miech\* Pauline Luc\* Iain Barr† Yana Hasson Karel Lenc† Arthur Menschi Katie Millican† Malcolm Reynolds<sup>†</sup> Roman Ring† Eliza Rutherford Serkan Cabi Tengda Han **Zhitao Gong** Sina Samangooei **Marianne Monteiro** Jacob Menick Sebastian Borgeaud Andrew Brock Aida Nematzadeh Sahand Sharifzadeh Mikolaj Binkowski Ricardo Barreira **Oriol Vinyals Andrew Zisserman**" and "Karen Simonyan\*,‡" (Title header)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

We introduce Flamingo, "a family of Visual Language Models (VLM) with this ability," to "rapidly adapt to a variety of image and video tasks" via in-context few-shot learning from interleaved visual-text inputs (Abstract).

## 3. Tasks Evaluated

The paper states it evaluates "16 multimodal image/video and language benchmarks" (Appendix B.1.4, Evaluation benchmarks).

| Task name | Task type | Dataset(s) used | Domain | Evidence (quotes) |
| --- | --- | --- | --- | --- |
| ImageNet-1k | Classification | ImageNet-1k | Image | "ImageNet-1k [94]" and "Object classification" in the "Image" row (Table 6, Appendix B.1.4). |
| MS-COCO | Generation | MS-COCO | Image | "MS-COCO [15]" and "Scene description" in the "Image" row (Table 6, Appendix B.1.4). |
| VQAv2 | Other (specify: question answering) | VQAv2 | Image | "VQAv2 [3]" and "Scene understanding QA" in the "Image" row (Table 6, Appendix B.1.4). |
| OKVQA | Other (specify: question answering) | OKVQA | Image | "OKVQA [69]" and "External knowledge QA" in the "Image" row (Table 6, Appendix B.1.4). |
| Flickr30k | Generation | Flickr30k | Image | "Flickr30k [139]" and "Scene description" in the "Image" row (Table 6, Appendix B.1.4). |
| VizWiz | Other (specify: question answering) | VizWiz | Image | "VizWiz [35]" and "Scene understanding QA" in the "Image" row (Table 6, Appendix B.1.4). |
| TextVQA | Other (specify: question answering) | TextVQA | Image | "TextVQA [100]" and "Text reading QA" in the "Image" row (Table 6, Appendix B.1.4). |
| VisDial | Other (specify: visual dialogue) | VisDial | Image | "VisDial [20]" and "Visual Dialogue" in the "Image" row (Table 6, Appendix B.1.4). |
| HatefulMemes | Classification | HatefulMemes | Image | "HatefulMemes [54]" and "Meme classification" in the "Image" row (Table 6, Appendix B.1.4). |
| Kinetics700 2020 | Classification | Kinetics700 2020 | Video | "Kinetics700 2020 [102]" and "Action classification" in the "Video" row (Table 6, Appendix B.1.4). |
| VATEX | Generation | VATEX | Video | "VATEX [122]" and "Event description" in the "Video" row (Table 6, Appendix B.1.4). |
| MSVDQA | Other (specify: question answering) | MSVDQA | Video | "MSVDQA [130]" and "Event understanding QA" in the "Video" row (Table 6, Appendix B.1.4). |
| YouCook2 | Generation | YouCook2 | Video | "YouCook2 [149]" and "Event description" in the "Video" row (Table 6, Appendix B.1.4). |
| MSRVTTQA | Other (specify: question answering) | MSRVTTQA | Video | "MSRVTTQA [130]" and "Event understanding QA" in the "Video" row (Table 6, Appendix B.1.4). |
| iVQA | Other (specify: question answering) | iVQA | Video | "iVQA [135]" and "Event understanding QA" in the "Video" row (Table 6, Appendix B.1.4). |
| RareAct | Other (specify: retrieval) | RareAct | Video | "RareAct [73]" and "Composite action retrieval" in the "Video" row (Table 6, Appendix B.1.4). |
| NextQA | Other (specify: question answering) | NextQA | Video | "NextQA [129]" and "Temporal/Causal QA" in the "Video" row (Table 6, Appendix B.1.4). |
| STAR | Other (specify: question answering, multiple-choice) | STAR | Video | "STAR [128]" and "Multiple-choice QA" in the "Video" row (Table 6, Appendix B.1.4). |

## 4. Domain and Modality Scope

- Evaluation scope: Multiple modalities and domains; the model ingests "images or videos" with text and is evaluated on "multimodal image/video and language benchmarks" (Abstract; Appendix B.1.4, Evaluation benchmarks).
- Single domain? No; the paper evaluates both image and video benchmarks ("image and video tasks") (Abstract).
- Multiple domains within the same modality? Yes; the evaluation spans multiple image and multiple video benchmarks ("image/video and language benchmarks") (Appendix B.1.4, Evaluation benchmarks).
- Multiple modalities? Yes; "images or videos" interleaved with text (Abstract).
- Does the paper claim domain generalization or cross-domain transfer? Not claimed.

## 5. Model Sharing Across Tasks

Few-shot evaluation uses shared weights: "A single Flamingo model reaches the state of the art on a wide array of image (I) and video (V) understanding tasks with few-shot learning" and does so "without adapting any model weights" (Table 1). The paper also reports per-task fine-tuning: "we explore fine-tuning our largest model, *Flamingo*, for a given task" and name five tasks in that context: "VQAv2, VATEX, VizWiz, MSRVTTQA, and HatefulMemes" (Section 3.2).

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| ImageNet-1k | Not specified | Not specified | Not specified | "ImageNet-1k [94]" is listed among benchmarks (Table 6, Appendix B.1.4). |
| MS-COCO | Yes (few-shot) | Not specified (few-shot uses no weight adaptation) | Not specified | "without adapting any model weights" (Table 1). |
| VQAv2 | Yes (few-shot) | Yes (reported) | Not specified | "without adapting any model weights" (Table 1); "VQAv2" listed among fine-tuned tasks (Section 3.2). |
| OKVQA | Yes (few-shot) | Not specified (few-shot uses no weight adaptation) | Not specified | "without adapting any model weights" (Table 1). |
| Flickr30k | Yes (few-shot) | Not specified (few-shot uses no weight adaptation) | Not specified | "without adapting any model weights" (Table 1). |
| VizWiz | Yes (few-shot) | Yes (reported) | Not specified | "without adapting any model weights" (Table 1); "VizWiz" listed among fine-tuned tasks (Section 3.2). |
| TextVQA | Yes (few-shot) | Not specified (few-shot uses no weight adaptation) | Not specified | "without adapting any model weights" (Table 1). |
| VisDial | Yes (few-shot) | Not specified (few-shot uses no weight adaptation) | Not specified | "without adapting any model weights" (Table 1). |
| HatefulMemes | Yes (few-shot) | Yes (reported) | Not specified | "without adapting any model weights" (Table 1); "HatefulMemes" listed among fine-tuned tasks (Section 3.2). |
| Kinetics700 2020 | Not specified | Not specified | Not specified | "Kinetics700 2020 [102]" is listed among benchmarks (Table 6, Appendix B.1.4). |
| VATEX | Yes (few-shot) | Yes (reported) | Not specified | "without adapting any model weights" (Table 1); "VATEX" listed among fine-tuned tasks (Section 3.2). |
| MSVDQA | Yes (few-shot) | Not specified (few-shot uses no weight adaptation) | Not specified | "without adapting any model weights" (Table 1). |
| YouCook2 | Yes (few-shot) | Not specified (few-shot uses no weight adaptation) | Not specified | "without adapting any model weights" (Table 1). |
| MSRVTTQA | Yes (few-shot) | Yes (reported) | Not specified | "without adapting any model weights" (Table 1); "MSRVTTQA" listed among fine-tuned tasks (Section 3.2). |
| iVQA | Yes (few-shot) | Not specified (few-shot uses no weight adaptation) | Not specified | "without adapting any model weights" (Table 1). |
| RareAct | Yes (few-shot) | Not specified (few-shot uses no weight adaptation) | Not specified | "without adapting any model weights" (Table 1). |
| NextQA | Yes (few-shot) | Not specified (few-shot uses no weight adaptation) | Not specified | "without adapting any model weights" (Table 1). |
| STAR | Yes (few-shot) | Not specified (few-shot uses no weight adaptation) | Not specified | "without adapting any model weights" (Table 1). |

## 6. Input and Representation Constraints

- Input resolution (fixed/variable): "The visual inputs are resized to  $320 \times 320$  while preserving their aspect ratios, padding the image with the mean value if required" (Appendix B.1.2). The vision pipeline also accepts "a variable number of image or video features" (Section 2.1).
- Fixed patch size: Not specified.
- Fixed number of tokens/frames: "produces a fixed number of visual outputs (64)" (Section 2.1); "sample a random subsequence of L=256 tokens and take up to the first N=5 images" (Section 2.4); "trained with a fixed number of 8 frames" (Appendix B.1.2).
- Fixed dimensionality: "a 2D spatial grid of features that is flattened to a 1D sequence" and for video "a 3D spatio-temporal grid of features" that is "flattened to 1D" (Section 2.1).
- Padding/resizing requirements: "padding the image with the mean value if required" during resizing (Appendix B.1.2).

## 7. Context Window and Attention Structure

- Maximum sequence length: "maximum sequence length (2048) our LMs have been trained on" (Discussion, Legacies of language models).
- Fixed or variable sequence length: Training samples use "a random subsequence of L=256 tokens" (Section 2.4), while evaluation prompts can be much longer ("prompt length ranges from 4096 to 8192 tokens") and include "up to 32 pairs (or \"shots\") of images/videos and corresponding texts" (Discussion, Legacies of language models; Section 2.3).
- Attention type: The model uses cross-attention and self-attention with masking: "masking the full text-to-image cross-attention matrix" and relying on "self-attention in the LM" (Section 2.3).
- Mechanisms to manage computational cost: The Perceiver Resampler "produces a fixed number of visual outputs (64), reducing the computational complexity of the vision-text cross-attention" and the model uses a single-image cross-attention mask (Section 2.1; Section 2.3).

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: For video features, "learned temporal embeddings are added" to the 3D spatio-temporal grid (Section 2.1). At inference time they use "linearly interpolating the learnt temporal position embedding of the Perceiver Resampler" (Appendix B.1.2). They also explored "learning absolute index embeddings added to the cross-attention features for each image" (Appendix B.3.1).
- Where it is applied: Temporal embeddings are added to video features (Section 2.1), and temporal position embeddings are part of the Perceiver Resampler (Appendix B.1.2). The explored absolute index embeddings are added to cross-attention features for each image (Appendix B.3.1).
- Fixed across experiments vs modified: The paper describes exploratory alternatives for image indexing ("learning absolute index embeddings") which were "not as robust" (Appendix B.3.1). No other positional encoding changes are specified.

## 9. Positional Encoding as a Variable

- Core variable or fixed assumption? Positional encoding is not a core research variable; it appears only in ablations about image indexing.
- Multiple positional encodings compared? Yes, they "explored more explicit ways" including "learning absolute index embeddings added to the cross-attention features for each image" (Appendix B.3.1).
- PE choice claimed as not critical? Not claimed; instead the explored strategies were "not as robust" (Appendix B.3.1).

## 10. Evidence of Constraint Masking

- Model sizes: Experiments span "the 1.4B, 7B, and 70B parameter Chinchilla models" used to build "Flamingo-3B, Flamingo-9B and Flamingo-80B" (Section 2.2).
- Scaling effects: "the larger the model, the better the few-shot performance" and "performance also improves with the number of shots" (Section 3.3, Scaling with respect to parameters and shots).
- Dataset scale: M3W is built from "approximately 43 million webpages" (Section 2.4). ALIGN has "1.8 billion images"; LTIP has "312 million image and text pairs"; VTP has "27 million short videos" (Section 2.4).
- Performance attribution to data mixture: "removing the interleaved image-text dataset M3W leads to a decrease of more than 17% in performance" and removing video-text data "negatively affects performance on all video tasks" (Section 3.3, Importance of the training data mixture).
- Training/architecture tricks: The model relies on GATED XATTN-DENSE layers and tanh gating; removing gating yields performance drops ("drop of 4.2%") (Section 3.3, Visual conditioning of the frozen LM).

## 11. Architectural Workarounds

- Perceiver Resampler to compress variable visual features: It "produces a fixed number of visual outputs (64), reducing the computational complexity of the vision-text cross-attention" (Section 2.1).
- GATED XATTN-DENSE layers to integrate vision into a frozen LM: "insert new cross-attention layers between existing pretrained and frozen LM layers" with gating (Section 2.2; Figure 4).
- Single-image cross-attention masking: "masking the full text-to-image cross-attention matrix" so each text token attends only to the most recent image (Section 2.3).
- Frozen pretrained backbones: The model "freeze[s] the pretrained LM blocks" and uses a pretrained vision encoder (Section 2.2; Section 2.1).

## 12. Explicit Limitations and Non-Claims

- Limitations from the paper: "LMs generalise poorly to sequences longer than the training ones" (Discussion, Limitations). The paper also notes that "the classification performance of Flamingo lags behind that of state-of-the-art contrastive models" (Discussion, Limitations).
- Few-shot limitations: "in-context learning is known to be highly sensitive to various aspects of the demonstrations" and its "inference compute cost and absolute performance scale poorly with the number of shots beyond this low-data regime" (Discussion, Limitations).
- Non-claims about domain generalization: No explicit claim of domain generalization or cross-domain transfer is stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Image and video benchmarks with text prompts; multiple modalities but fixed benchmark suites.
> - Task structure: Predefined tasks (classification, captioning, QA, visual dialogue) framed as text prediction with few-shot prompts.
> - Representation rigidity: Fixed 320x320 resizing and padding, fixed 64 visual tokens, and fixed 8-frame training clips; token-length sampling (L=256, N=5) during M3W training.
> - Model sharing vs specialization: Single shared model weights for few-shot evaluation; per-task fine-tuning reported for a subset of tasks.
> - Role of positional encoding: Temporal position embeddings for video features; image index embeddings only explored as a non-robust ablation.

### 14. Final Classification

**Multi-task, multi-domain (constrained).** The paper evaluates "16 multimodal image/video and language benchmarks" spanning image and video tasks (Appendix B.1.4), and emphasizes that a "single Flamingo model" is used across tasks "without adapting any model weights" in few-shot settings (Table 1). The tasks are fixed, benchmarked datasets with prescribed prompts rather than open-ended multi-domain learning, making the setup multi-task/multi-domain but constrained.
