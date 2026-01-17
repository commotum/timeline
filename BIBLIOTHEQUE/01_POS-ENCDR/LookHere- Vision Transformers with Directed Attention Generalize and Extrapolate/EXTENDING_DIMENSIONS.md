## 1. Basic Metadata

- Title: "LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate" (Title)
- Authors: "Anthony Fuller, Daniel G. Kyrollos, Yousef Yassin, James R. Green" (Authors)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

It addresses that "ViTs poorly extrapolate to more patches at test time" by proposing LookHere, "a drop-in replacement for the position encoding of plain ViTs that restricts attention heads to fixed fields of view, pointed in different directions, using 2D attention masks" (Abstract).

## 3. Tasks Evaluated

- Task name: Image classification (standard ImageNet test sets); Task type: Classification; Dataset(s): ImageNet (Val, ReaL, v2, -A, -R, -HR); Domain: images; Quotes: "We demonstrate that LookHere improves performance on classification (avg. \( \gamma \) 1.6\%), against adversarial attack (avg. $\uparrow 5.4\%$ ), and decreases calibration error (avg. $\downarrow 1.5\%$ ) — on ImageNet without extrapolation." (Abstract) "We test all 80 models on six ImageNet test sets. This includes ① the original \"validation\" set used as a test set (Val for short [1]), ② the reassessed labels of the original validation set (ReaL for short [4]), ③ the independently collected and in-distribution test set (v2 for short [2]), ④ the natural adversarial test set (-A for short [3]), ⑤ the ImageNet rendition test set (-R for short [5]), and ⑥ the high-resolution test set that we introduce (-HR for short)." (Section 4.1 Setup)
- Task name: Image classification (resolution extrapolation); Task type: Classification; Dataset(s): ImageNet (same test sets); Domain: images; Quotes: "Extrapolating. With the best model per method, we test on images larger than  $224^2$  px, increasing the number of patches and we test on images smaller than  $224^2$  px, decreasing the number of patches; for both experiments, no further training is performed — the models are tested on their resolution generalization ability." (Section 4.1 Setup)
- Task name: Image classification (high-resolution finetuning); Task type: Classification; Dataset(s): ImageNet; Domain: images; Quotes: "With the best model per method, we continue training on ImageNet for 5 epochs at  $384^2$  px. We test at  $384^2$  px without extrapolating." (Section 4.1 Setup)
- Task name: Adversarial robustness evaluation (FGSM); Task type: Classification; Dataset(s): ImageNet Val; Domain: images; Quotes: "We perform Fast Gradient Sign Method (FGSM [82]) adversarial attacks with two strengths  $(\frac{1}{255}, \frac{3}{255})$  on all models using Val images." (Section 4.1 Setup)
- Task name: Calibration estimation (ECE); Task type: Classification; Dataset(s): ImageNet Val; Domain: images; Quotes: "We calculate the Expected Calibration Error (ECE [83]) with 15 bins of all models using Val images." (Section 4.1 Setup)
- Task name: Semantic segmentation (finetuning); Task type: Segmentation; Dataset(s): ADE20k, Cityscapes; Domain: images; Quotes: "Segmentation. With the best model per method, we finetune following the Segmenter protocol with a linear decoder [84]." (Section 4.1 Setup) "We run these experiments on ADE20k [86] at 512<sup>2</sup> px and Cityscapes [87] at 768<sup>2</sup> px." (Section 4.1 Setup)
- Task name: Semantic segmentation (linear probing); Task type: Segmentation; Dataset(s): ADE20k, Cityscapes; Domain: images; Quotes: "Additionally, we probe the patches by only training a linear layer to produce a low-resolution logit map which is upsampled to obtain a full resolution segmentation map, following [85]." (Section 4.1 Setup)
- Task name: Patch logit-lens segmentation probing; Task type: Segmentation; Other (representation probing); Dataset(s): ImageNet-S; Domain: images; Quotes: "Patch Logit-lens. Inspired by interpretability research [88], we evaluate the quality of the learned patch representations for models leveraging LookHere compared with other methods." (Section 4.1 Setup) "We leverage the ImageNet-S dataset [91], which contains partial segmentation maps for 12k images from Val, covering 919 ImageNet classes." (Section 4.1 Setup)

## 4. Domain and Modality Scope

- Evaluation scope: Multiple datasets within the same modality (images), including ImageNet and segmentation datasets. Evidence: "We test all 80 models on six ImageNet test sets." (Section 4.1 Setup) "We run these experiments on ADE20k [86] at 512<sup>2</sup> px and Cityscapes [87] at 768<sup>2</sup> px." (Section 4.1 Setup)
- Multiple modalities?: Not stated; the paper frames the problem in images ("High-resolution images offer more information about scenes that can improve model accuracy." (Abstract)).
- Domain generalization or cross-domain transfer?: Resolution generalization within images is claimed ("the models are tested on their resolution generalization ability." (Section 4.1 Setup)); cross-domain transfer is not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Image classification (standard ImageNet test sets) | Yes | No | No | "We train all models from scratch for 150 epochs on  $224^2$  px images." (Section 4.1 Setup); "We test all 80 models on six ImageNet test sets." (Section 4.1 Setup) |
| Image classification (resolution extrapolation) | Yes | No | No | "Extrapolating. With the best model per method, we test on images larger than  $224^2$  px, increasing the number of patches and we test on images smaller than  $224^2$  px, decreasing the number of patches; for both experiments, no further training is performed — the models are tested on their resolution generalization ability." (Section 4.1 Setup) |
| Image classification (high-resolution finetuning) | Yes | Yes | No | "With the best model per method, we continue training on ImageNet for 5 epochs at  $384^2$  px. We test at  $384^2$  px without extrapolating." (Section 4.1 Setup) |
| Adversarial robustness evaluation (FGSM) | Yes | No | No | "We perform Fast Gradient Sign Method (FGSM [82]) adversarial attacks with two strengths  $(\frac{1}{255}, \frac{3}{255})$  on all models using Val images." (Section 4.1 Setup) |
| Calibration estimation (ECE) | Yes | No | No | "We calculate the Expected Calibration Error (ECE [83]) with 15 bins of all models using Val images." (Section 4.1 Setup) |
| Semantic segmentation (finetuning) | Yes | Yes | Yes (linear decoder) | "With the best model per method, we finetune following the Segmenter protocol with a linear decoder [84]." (Section 4.1 Setup) |
| Semantic segmentation (linear probing) | Yes | No (linear layer only) | Yes (linear layer) | "we probe the patches by only training a linear layer to produce a low-resolution logit map which is upsampled to obtain a full resolution segmentation map, following [85]." (Section 4.1 Setup) |
| Patch logit-lens segmentation probing | Yes | No | No (uses learned classifier head) | "we project frozen patch representations onto the learned class embedding space using the MLP classifier head that was learned for the CLS token." (Section 4.1 Setup) |

## 6. Input and Representation Constraints

- 2D image grid of non-overlapping patches: "A ViT splits an image into a grid of non-overlapping patches, flattens the grid into a sequence, and flattens the patches into vectors; i.e.,  $\mathbb{R}^{Y \times X \times C} \to \mathbb{R}^{N_y \times N_x \times P^2 \times C} \to \mathbb{R}^{(N_y \cdot N_x) \times (P^2 \cdot C)}$" (Section 2 Background and Related Work)
- Sequence length varies with number of patches: "ViTs poorly extrapolate to more patches at test time" (Abstract); "we test on images larger than  $224^2$  px, increasing the number of patches and we test on images smaller than  $224^2$  px, decreasing the number of patches" (Section 4.1 Setup)
- Training resolution fixed per run: "We train all models from scratch for 150 epochs on  $224^2$  px images." (Section 4.1 Setup)
- High-resolution finetuning uses fixed target size: "we continue training on ImageNet for 5 epochs at  $384^2$  px." (Section 4.1 Setup)
- Cropping/resizing for ImageNet-HR: "We manually collect 5 images for each ImageNet class, resulting in 5k total images, and manually crop them to  $1024^2$  px." (Section 4.1 Setup)
- Fixed patch size: Not specified.
- Fixed number of tokens: Not specified (number of patches explicitly varies).
- Padding requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; maximum evaluation resolution stated as "tested up to  $1024^2$  px." (Figure 1)
- Sequence length fixed or variable: Variable, with explicit testing at larger and smaller resolutions: "we test on images larger than  $224^2$  px, increasing the number of patches and we test on images smaller than  $224^2$  px, decreasing the number of patches" (Section 4.1 Setup)
- Attention type: Sparse/masked directional attention: "restricts attention heads to fixed fields of view (FOV) and points them in different directions via 2D masks" (Section 1 Introduction); "preventing attention outside the head's FOV" (Section 3 Design Motivation). Some heads remain global: "we leave the last four attention heads undirected to allow them unrestricted attention over the full image." (Section 3)
- Computational cost mechanisms: "LookHere matrices offer structured sparsity (up to 7/8 for a  $45^{\circ}$  FOV) that can speedup attention" (Compute)

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used (LookHere): "We introduce a novel position encoding method for plain ViTs that restricts attention heads to fixed fields of view (FOV) and points them in different directions via 2D masks." (Section 1 Introduction) "We encode positions by subtracting the LookHere matrix for a layer l,  $\mathcal{A}_{FIX}^l$ , from the learned attention matrix,  $\mathcal{A}_{LRN}^l = QK^T/\sqrt{D_H}$ , before the softmax that normalizes the attention matrix prior to multiplying it by values [75], i.e.,  $A^l = \text{softmax}(A_{LRN}^l - A_{FIX}^l)$ ." (Section 3 Design Specifics) and "We do not add position embeddings to patch embeddings." (Section 3 Design Specifics)
- Baseline positional encodings compared: "Input Embeddings. This group leverages learned or fixed position embeddings,  $E_i^{pos} \in \mathbb{R}^D$ , that are added to patch embeddings at the transformer input" (Section 2) and "Attention Biases. This group leverages learned or fixed operations that encode positions by modifying the pairwise interactions between patches in self-attention *without* adding position embeddings to patch embeddings." (Section 2)
- Where applied: LookHere modifies attention every layer: "The final attention matrix is computed as  $\mathcal{A}^l = \mathtt{softmax}(\mathcal{A}^l_{LRN} - \mathcal{A}^l_{FIX})$ , at each layer l." (Figure 2). Input-embedding baselines apply at input: "added to patch embeddings at the transformer input" (Section 2)
- Fixed across experiments or compared/ablated: Multiple encodings compared and ablated: "We perform an apples-to-apples comparison between *seven* position encoding methods for plain ViTs alongside our three LookHere variants." (Section 1 Introduction) "Design Ablations. We offer four takeaways through extensive ablations" (Section 3)

## 9. Positional Encoding as a Variable

- Core research variable: Yes. Evidence: "We introduce a novel position encoding method for plain ViTs" and "We perform an apples-to-apples comparison between *seven* position encoding methods for plain ViTs alongside our three LookHere variants." (Section 1 Introduction)
- Multiple positional encodings compared: Yes (same evidence as above).
- PE claimed "not critical" or secondary: Not stated.

## 10. Evidence of Constraint Masking

- Model sizes: "ViT-B/16 models trained for 150 epochs on ImageNet" (Figure 1). Limitations emphasize size choice: "we select the most common size, the ViT-B/16" (Limitations)
- Dataset sizes: "we train a ViT-B/16 on 99% of the ImageNet training set, holding the last 1% as a validation set" (Section 4.1 Setup); "We manually collect 5 images for each ImageNet class, resulting in 5k total images" (Section 4.1 Setup)
- Performance gains attributed to architecture rather than scaling: "We introduce a novel position encoding method for plain ViTs" and "We perform an apples-to-apples comparison between *seven* position encoding methods" (Section 1 Introduction); scaling is explicitly limited: "The primary limitation of our experiments is we do not scale ViTs to giant sizes." (Limitations)
- Training tricks/hyperparameter control: "We search 8 hyperparameter configurations for *each* position encoding method" (Section 4 Experiments)

## 11. Architectural Workarounds

- Directional attention masks with FOV constraints: "We introduce 2D attention masks that assign each attention head a direction and a FOV, preventing attention outside the head's FOV." (Section 3 Design Motivation)
- Distance-based attention penalties within FOV: "Within a head's FOV, attention scores are penalized based on relative patch distances." (Section 3 Design Motivation)
- Bias-based positional encoding instead of input embeddings: "We encode positions by subtracting the LookHere matrix for a layer l,  $\mathcal{A}_{FIX}^l$ , from the learned attention matrix,  $\mathcal{A}_{LRN}^l = QK^T/\sqrt{D_H}$ , before the softmax that normalizes the attention matrix prior to multiplying it by values [75], i.e.,  $A^l = \text{softmax}(A_{LRN}^l - A_{FIX}^l)$ ." (Section 3 Design Specifics) and "We do not add position embeddings to patch embeddings." (Section 3 Design Specifics)
- Mixed global and constrained heads: "we leave the last four attention heads undirected to allow them unrestricted attention over the full image." (Section 3)
- Structured sparsity for efficiency: "LookHere matrices offer structured sparsity (up to 7/8 for a  $45^{\circ}$  FOV) that can speedup attention" (Compute)
- Fixed grid assumption: "A ViT splits an image into a grid of non-overlapping patches" (Section 2 Background and Related Work)

## 12. Explicit Limitations and Non-Claims

- Limitations: "The primary limitation of LookHere is it requires hand-designed directional masks and distance penalties." (Limitations) "The primary limitation of our experiments is we do not scale ViTs to giant sizes." (Limitations)
- Future work: "We are excited to realize the computational gains that LookHere makes available via sparse attention kernels, as well as bring LookHere to video and 3D point-cloud applications." (Future Work)
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: multiple image datasets (ImageNet, ADE20k, Cityscapes) within a single modality.
> – Task structure: classification plus robustness/calibration and segmentation; resolution extrapolation tests stay within images.
> – Representation rigidity: fixed 2D patch grid; training at fixed resolutions with evaluation at larger/smaller sizes.
> – Model sharing vs specialization: same ViT-B/16 backbone reused across evaluations; fine-tuning for high-res and segmentation; linear probes for patch analysis.
> – Role of positional encoding: central experimental variable with multiple PE methods; LookHere uses attention masks/biases instead of input embeddings.

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple tasks on images, including "classification," "Adversarial Attacks," "Calibration Estimates," and "Segmentation" across ImageNet, ADE20k, and Cityscapes (Abstract; Section 4.1 Setup). It tests resolution generalization within the same modality ("Extrapolating. With the best model per method, we test on images larger than  $224^2$  px, increasing the number of patches and we test on images smaller than  $224^2$  px, decreasing the number of patches; for both experiments, no further training is performed — the models are tested on their resolution generalization ability.") and does not claim cross-domain or multi-modality transfer (Section 4.1 Setup).
