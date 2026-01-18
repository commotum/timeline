## 1. Basic Metadata

Title: "ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision" (Title)
Authors: "Wonjae Kim\*1† Bokyung Son\*1 Ildoo Kim²" (Title block)
Year: "Proceedings of the 38<sup>th</sup> International Conference on Machine Learning, PMLR 139, 2021." (Front matter)
Venue (conference/journal/arXiv): "Proceedings of the 38<sup>th</sup> International Conference on Machine Learning, PMLR 139" (Front matter)

## 2. One-Sentence Contribution Summary

"In this paper, we present a minimal VLP model, Vision-and-Language Transformer (ViLT), monolithic in the sense that the processing of visual inputs is drastically simplified to just the same convolution-free manner that we process textual inputs." (Abstract)

## 3. Tasks Evaluated

Task name: Visual Question Answering (VQAv2)
Task type: Classification
Dataset(s) used: VQAv2
Domain: "pairs of an image and a question in natural language" (Section 4.3 Classification Tasks)
Evidence: "We evaluate ViLT on two widely explored types of visionand-language downstream tasks: for *classification*, we use VQAv2 (Goyal et al., 2017) and NLVR2 (Suhr et al., 2018), and for *retrieval*, we use MSCOCO and Flickr30K (F30K) (Plummer et al., 2015) re-splited by Karpathy & Fei-Fei (2015)." (Section 4.1 Overview); "The VQAv2 task asks for answers given pairs of an image and a question in natural language." (Section 4.3 Classification Tasks)

Task name: Natural Language for Visual Reasoning (NLVR2)
Task type: Classification
Dataset(s) used: NLVR2
Domain: "triplets of two images and a question in natural language" (Section 4.3 Classification Tasks)
Evidence: "We evaluate ViLT on two widely explored types of visionand-language downstream tasks: for *classification*, we use VQAv2 (Goyal et al., 2017) and NLVR2 (Suhr et al., 2018), and for *retrieval*, we use MSCOCO and Flickr30K (F30K) (Plummer et al., 2015) re-splited by Karpathy & Fei-Fei (2015)." (Section 4.1 Overview); "The NLVR2 task is a binary classification task given triplets of two images and a question in natural language." (Section 4.3 Classification Tasks)

Task name: Image-text retrieval (image-to-text and text-to-image)
Task type: Other (retrieval)
Dataset(s) used: MSCOCO; Flickr30K (F30K)
Domain: "image-to-text and text-to-image retrieval" (Section 4.4 Retrieval Tasks)
Evidence: "We evaluate ViLT on two widely explored types of visionand-language downstream tasks: for *classification*, we use VQAv2 (Goyal et al., 2017) and NLVR2 (Suhr et al., 2018), and for *retrieval*, we use MSCOCO and Flickr30K (F30K) (Plummer et al., 2015) re-splited by Karpathy & Fei-Fei (2015)." (Section 4.1 Overview); "For image-to-text and text-to-image retrieval, we measure both zero-shot and fine-tuned performance<sup>8</sup>." (Section 4.4 Retrieval Tasks)

## 4. Domain and Modality Scope

Single domain? Evaluation is on vision-and-language tasks with image and text inputs: "vision-and-language downstream tasks where the inputs involve two modalities." (Introduction)
Multiple domains within the same modality? Not explicitly stated; all evaluated datasets are within vision-and-language tasks ("We evaluate ViLT on two widely explored types of visionand-language downstream tasks: for *classification*, we use VQAv2 (Goyal et al., 2017) and NLVR2 (Suhr et al., 2018), and for *retrieval*, we use MSCOCO and Flickr30K (F30K) (Plummer et al., 2015) re-splited by Karpathy & Fei-Fei (2015).") (Section 4.1 Overview).
Multiple modalities? Yes — "the inputs involve two modalities." (Introduction)
Does the paper claim domain generalization or cross-domain transfer? Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| VQAv2 (VQA) | Yes | Yes | Yes | "We pre-train ViLT-B/32 for 100K or 200K steps on 64 NVIDIA V100 GPUs with a batch size of 4,096." (Section 4.2 Implementation Details); "Following this practice, we fine-tune ViLT-B/32 on the VQAv2 train and validation sets while reserving 1,000 validation images and their related questions for internal validation." (Section 4.3 Classification Tasks); "We use a two-layer MLP of hidden size 1,536 as the fine-tuned downstream head." (Section 4.3 Classification Tasks) |
| NLVR2 | Yes | Yes | Yes | "We pre-train ViLT-B/32 for 100K or 200K steps on 64 NVIDIA V100 GPUs with a batch size of 4,096." (Section 4.2 Implementation Details); "The NLVR2 task is a binary classification task given triplets of two images and a question in natural language." (Section 4.3 Classification Tasks); "The head takes the concatenation of two pooled representations (p) as input and outputs the binary prediction." (Section 4.3 Classification Tasks) |
| Image-text retrieval (MSCOCO, F30K) | Yes | Yes | Yes | "We pre-train ViLT-B/32 for 100K or 200K steps on 64 NVIDIA V100 GPUs with a batch size of 4,096." (Section 4.2 Implementation Details); "We fine-tune ViLT-B/32 on the Karpathy & Fei-Fei (2015) split of MSCOCO and F30K." (Section 4.4 Retrieval Tasks); "We initialize the similarity score head from" and "the pre-trained ITM head, particularly the part that computes the true-pair logits." (Section 4.4 Retrieval Tasks) |

## 6. Input and Representation Constraints

- "We resize the shorter edge of input images to 384 and limit the longer edge to under 640 while preserving the aspect ratio." (Section 4.2 Implementation Details)
- "We use a  $32 \times 32$  patch projection which only requires 2.4M parameters." (Section 2.3 Visual Embedding Schema)
- "For all experiments, we use weights from ViT-B/32 pretrained on ImageNet, hence the name ViLT-B/32.<sup>5</sup> Hidden size H is 768, layer depth D is 12, patch size P is 32, MLP size is 3,072, and the number of attention heads is 12." (Section 3.1 Model Overview)
- "The input image  $I \in \mathbb{R}^{C \times H \times W}$  is sliced into patches and flattened to  $v \in \mathbb{R}^{N \times (P^2 \cdot C)}$  where (P,P) is the patch resolution and  $N = HW/P^2$ ." (Section 3.1 Model Overview)
- "Patch projection of ViLT-B/32 yields 12  $\times$  20 = 240 patches for an image with a resolution of 384  $\times$  640. As this is a rarely reached upper limit, we sample 200 patches at maximum during pre-training." (Section 4.2 Implementation Details)
- "We interpolate  $V^{\rm pos}$  of ViT-B/32 to fit the size of each image and pad the patches for batch training." (Section 4.2 Implementation Details)
- "The input text  $t \in \mathbb{R}^{L \times |V|}$  is embedded to  $\bar{t} \in \mathbb{R}^{L \times H}$  with a word embedding matrix  $T \in \mathbb{R}^{|V| \times H}$  and a position embedding matrix  $T^{\text{pos}} \in \mathbb{R}^{(L+1) \times H}$ ." (Section 3.1 Model Overview)

## 7. Context Window and Attention Structure

Maximum sequence length: "Patch projection of ViLT-B/32 yields 12  $\times$  20 = 240 patches for an image with a resolution of 384  $\times$  640. As this is a rarely reached upper limit, we sample 200 patches at maximum during pre-training." (Section 4.2 Implementation Details). Text length is not specified beyond "The input text  $t \in \mathbb{R}^{L \times |V|}$" (Section 3.1 Model Overview).
Fixed or variable sequence length: Variable, since "N = HW/P^2" and patches are padded (Section 3.1 Model Overview; Section 4.2 Implementation Details: "pad the patches for batch training").
Attention type: Global self-attention in a single stream, as "ViT consists of stacked blocks that include a multiheaded self-attention (MSA) layer" and the modalities are "concatenated into a combined sequence  $z^0$ ." (Section 3.1 Model Overview).
Mechanisms to manage computational cost: Patch projection and sampling/padding of patches, e.g., "We use a  $32 \times 32$  patch projection" and "we sample 200 patches at maximum during pre-training" with resizing (Section 2.3 Visual Embedding Schema; Section 4.2 Implementation Details).

## 8. Positional Encoding (Critical Section)

- Mechanism: Learned position embedding matrices for text and image — "a position embedding matrix  $T^{\text{pos}} \in \mathbb{R}^{(L+1) \times H}$" and "position embedding  $V^{\mathrm{pos}} \in \mathbb{R}^{(N+1) \times H}$" (Section 3.1 Model Overview).
- Where applied: Added to input embeddings — "$\bar{t} = [t_{\text{class}}; t_1 T; \dots; t_L T] + T^{\text{pos}}$" and "$\bar{v} = [v_{\text{class}}; v_1 V; \dots; v_N V] + V^{\text{pos}}$" (Section 3.1 Model Overview, Eq. 1–2).
- Fixed across experiments or modified: "For all experiments, we use weights from ViT-B/32 pretrained on ImageNet, hence the name ViLT-B/32.<sup>5</sup> Hidden size H is 768, layer depth D is 12, patch size P is 32, MLP size is 3,072, and the number of attention heads is 12." and "We interpolate  $V^{\rm pos}$  of ViT-B/32 to fit the size of each image and pad the patches for batch training." (Section 3.1 Model Overview; Section 4.2 Implementation Details). No ablation or alternative positional encodings are described.

## 9. Positional Encoding as a Variable

Does the paper treat positional encoding as a core research variable? No; it is a fixed architectural component: "a position embedding matrix  $T^{\text{pos}} \in \mathbb{R}^{(L+1) \times H}$" and "position embedding  $V^{\mathrm{pos}} \in \mathbb{R}^{(N+1) \times H}$" (Section 3.1 Model Overview).
Are multiple positional encodings compared? Not stated.
Does the paper claim PE choice is "not critical" or secondary? Not stated.

## 10. Evidence of Constraint Masking

- Model size reported for ViLT-B/32: "Linear | ViLT-B/32200+40 | 87.4 | 55.9 | ~15" (Table 6).
- Dataset sizes for pre-training: "| MSCOCO           | 113K     | 567K       | $11.81 \pm 2.81$ |"; "| VG               | 108K     | 5.41M      | $5.53 \pm 1.76$  |"; "| GCC <sup>†</sup> | 3.01M    | 3.01M      | $10.66 \pm 4.93$ |"; "| SBU <sup>†</sup> | 867K     | 867K       | $15.0 \pm 7.74$  |" (Table 1).
- Training scale (steps): "We pre-train ViLT-B/32 for 100K or 200K steps" (Section 4.2 Implementation Details).
- Performance gains attributed to training tricks: "More training steps, whole word masking, and image augmentation come to be beneficial" (Section 4.5 Ablation Study).
- Data scaling claim: "the performance of pre-trained transformers scale well given an appropriate amount of data" (Section 5 Conclusion).

## 11. Architectural Workarounds

- Patch projection to reduce visual embedding overhead: "To minimize overhead, we adopt the simplest visual embedding scheme: *linear projection* that operates on image patches." (Section 2.3 Visual Embedding Schema)
- Single-stream interaction to avoid extra parameters: "We follow the single-stream approach for our interaction transformer module because the dual-stream approach introduces additional parameters." (Section 2.2 Modality Interaction Schema)
- Visual token budget and resizing: "We resize the shorter edge of input images to 384 and limit the longer edge to under 640" and "we sample 200 patches at maximum during pre-training" (Section 4.2 Implementation Details).
- NLVR2 pair strategy and head design: "we use the *pair* method" and "The head takes the concatenation of two pooled representations (p) as input and outputs the binary prediction." (Section 4.3 Classification Tasks)
- Task-specific heads: "We use a two-layer MLP of hidden size 1,536 as the fine-tuned downstream head."; "We initialize the similarity score head from" and "the pre-trained ITM head, particularly the part that computes the true-pair logits." (Section 4.3 Classification Tasks; Section 4.4 Retrieval Tasks).

## 12. Explicit Limitations and Non-Claims

- "Although remarkable as it is, ViLT-B/32 is more of a proof of concept that efficient VLP models free of convolution and region supervision can still be competent." (Section 5 Conclusion)
- "We leave training larger models for future work because aligned vision-and-language datasets are yet scarce." (Section 5 Conclusion)
- "However, MPP turns out not to be contributing to downstream performance" (Section 4.5 Ablation Study).
- "We stop increasing the number of iterations over 200K as the fine-tuned text retrieval performance decreases afterward." (Section 4.5 Ablation Study)

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: vision-and-language tasks with image and text inputs, e.g., "vision-and-language downstream tasks where the inputs involve two modalities" (Introduction) and classification/retrieval on VQAv2/NLVR2/MSCOCO/F30K (Section 4.1 Overview).
> – Task structure: classification and retrieval tasks only, "We evaluate ViLT on two widely explored types of visionand-language downstream tasks: for *classification*, we use VQAv2 (Goyal et al., 2017) and NLVR2 (Suhr et al., 2018), and for *retrieval*, we use MSCOCO and Flickr30K (F30K) (Plummer et al., 2015) re-splited by Karpathy & Fei-Fei (2015)." (Section 4.1 Overview).
> – Representation rigidity: resized images and fixed patch size with capped tokens ("We resize the shorter edge of input images to 384 and limit the longer edge to under 640 while preserving the aspect ratio."; "We use a  $32 \times 32$  patch projection which only requires 2.4M parameters."; "As this is a rarely reached upper limit, we sample 200 patches at maximum during pre-training.") (Section 4.2 Implementation Details; Section 2.3 Visual Embedding Schema).
> – Model sharing vs specialization: shared pre-trained backbone with task-specific heads, "We pre-train ViLT-B/32 for 100K or 200K steps on 64 NVIDIA V100 GPUs with a batch size of 4,096." and "We use a two-layer MLP of hidden size 1,536 as the fine-tuned downstream head." (Section 4.2 Implementation Details; Section 4.3 Classification Tasks).
> – Role of positional encoding: fixed learned position embeddings with interpolation, "The input text  $t \in \mathbb{R}^{L \times |V|}$  is embedded to  $\bar{t} \in \mathbb{R}^{L \times H}$  with a word embedding matrix  $T \in \mathbb{R}^{|V| \times H}$  and a position embedding matrix  $T^{\text{pos}} \in \mathbb{R}^{(L+1) \times H}$ ." and "We interpolate  $V^{\rm pos}$  of ViT-B/32 to fit the size of each image and pad the patches for batch training." (Section 3.1 Model Overview; Section 4.2 Implementation Details).

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple downstream tasks — "We evaluate ViLT on two widely explored types of visionand-language downstream tasks: for *classification*, we use VQAv2 (Goyal et al., 2017) and NLVR2 (Suhr et al., 2018), and for *retrieval*, we use MSCOCO and Flickr30K (F30K) (Plummer et al., 2015) re-splited by Karpathy & Fei-Fei (2015)." (Section 4.1 Overview) — which makes it multi-task. All tasks are vision-and-language with image and text inputs ("vision-and-language downstream tasks where the inputs involve two modalities") and no cross-domain transfer is claimed, so the domain scope is single (Introduction).
