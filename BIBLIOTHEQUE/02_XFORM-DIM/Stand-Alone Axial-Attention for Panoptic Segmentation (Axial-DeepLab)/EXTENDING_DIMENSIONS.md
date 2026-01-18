## 1. Basic Metadata

- Title: "Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation" (Title)
- Authors: "Huiyu Wang $^{1\star}$ , Yukun Zhu², Bradley Green², Hartwig Adam², Alan Yuille¹, and Liang-Chieh Chen²" (Author line)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper's primary contribution is to "attempt to remove this constraint by factorizing 2D self-attention into two 1D selfattentions" and to "propose a position-sensitive self-attention design" that yields axial-attention models for "image classification and dense prediction" (Abstract).

## 3. Tasks Evaluated

Task name: Image classification
Task type: Classification
Dataset(s) used: ImageNet
Domain: images (computer vision)
Quotes: "We show the effectiveness of our axial-attention models on ImageNet [70] for classification, and on three datasets (COCO [56], Mapillary Vistas [62], and Cityscapes [22]) for panoptic segmentation [45], instance segmentation, and semantic segmentation." (1 Introduction); "We conduct experiments on four large-scale datasets. We first report results with our Axial-ResNet on ImageNet [70]." (4 Experimental Results)

Task name: Panoptic segmentation
Task type: Segmentation
Dataset(s) used: COCO, Mapillary Vistas, Cityscapes
Domain: images (computer vision)
Quotes: "We then convert the ImageNet pretrained Axial-ResNet to Axial-DeepLab, and report results on COCO [56], Mapillary Vistas [62], and Cityscapes [22] for panoptic segmentation, evaluated by panoptic quality (PQ) [45]." (4 Experimental Results); "The heads produce semantic segmentation and class-agnostic instance segmentation, and they are merged by majority voting [89] to form the final panoptic segmentation." (3 Method)

Task name: Instance segmentation
Task type: Segmentation
Dataset(s) used: Mapillary Vistas, Cityscapes
Domain: images (computer vision)
Quotes: "We show the effectiveness of our axial-attention models on ImageNet [70] for classification, and on three datasets (COCO [56], Mapillary Vistas [62], and Cityscapes [22]) for panoptic segmentation [45], instance segmentation, and semantic segmentation." (1 Introduction); "We also report average precision (AP) for instance segmentation, and mean IoU for semantic segmentation on Mapillary Vistas and Cityscapes." (4 Experimental Results)

Task name: Semantic segmentation
Task type: Segmentation
Dataset(s) used: Mapillary Vistas, Cityscapes
Domain: images (computer vision)
Quotes: "We show the effectiveness of our axial-attention models on ImageNet [70] for classification, and on three datasets (COCO [56], Mapillary Vistas [62], and Cityscapes [22]) for panoptic segmentation [45], instance segmentation, and semantic segmentation." (1 Introduction); "We also report average precision (AP) for instance segmentation, and mean IoU for semantic segmentation on Mapillary Vistas and Cityscapes." (4 Experimental Results)

## 4. Domain and Modality Scope

- Evaluation scope: Multiple datasets within the same modality (images), since the paper reports results on "four large-scale datasets" and specifically "ImageNet [70]" plus "COCO [56], Mapillary Vistas [62], and Cityscapes [22]." (Abstract; 4 Experimental Results; 1 Introduction)
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Image classification | Not specified (single-task Axial-ResNet) | Not specified | Not specified | "We first report results with our Axial-ResNet on ImageNet [70]." (4 Experimental Results) |
| Panoptic segmentation | Yes (shared backbone with semantic/instance heads) | ImageNet-pretrained backbone converted to Axial-DeepLab (fine-tuning not explicitly stated) | Yes | "We then convert the ImageNet pretrained Axial-ResNet to Axial-DeepLab" (4 Experimental Results); "The heads produce semantic segmentation and class-agnostic instance segmentation, and they are merged by majority voting [89] to form the final panoptic segmentation." (3 Method) |
| Instance segmentation | Yes (shared backbone with panoptic pipeline) | ImageNet-pretrained backbone converted to Axial-DeepLab (fine-tuning not explicitly stated) | Yes | "The heads produce semantic segmentation and class-agnostic instance segmentation, and they are merged by majority voting [89] to form the final panoptic segmentation." (3 Method); "We then convert the ImageNet pretrained Axial-ResNet to Axial-DeepLab" (4 Experimental Results) |
| Semantic segmentation | Yes (shared backbone with panoptic pipeline) | ImageNet-pretrained backbone converted to Axial-DeepLab (fine-tuning not explicitly stated) | Yes | "The heads produce semantic segmentation and class-agnostic instance segmentation, and they are merged by majority voting [89] to form the final panoptic segmentation." (3 Method); "We then convert the ImageNet pretrained Axial-ResNet to Axial-DeepLab" (4 Experimental Results) |

## 6. Input and Representation Constraints

- Fixed or variable input resolution: Variable; "In cases where the inputs are extremely large  $(e.g., 2177 \times 2177)$  and memory is constrained" (3 Method, Axial-DeepLab).
- Fixed patch size: Not specified.
- Fixed number of tokens: Not specified.
- Fixed dimensionality (e.g., strictly 2D): "Given an input feature map  $x \in \mathbb{R}^{h \times w \times d_{in}}$  with height h, width w, and channels  $d_{in}$" (3.1 Position-Sensitive Self-Attention); "factorizing 2D self-attention into two 1D selfattentions" (Abstract).
- Any padding or resizing requirements: Not specified.
- Other representation constraints: "we extract feature maps with output stride (*i.e.*, the ratio of input resolution to the final backbone feature resolution) 16. We do not pursue output stride 8, since it is computationally expensive." (3 Method, Axial-DeepLab)

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; attention span can be global by setting "the span m directly to the whole input features." (3.2 Axial-Attention)
- Fixed or variable sequence length: Variable; "Optionally, one could also use a fixed m value" and global span is achieved by setting "the span m directly to the whole input features." (3.2 Axial-Attention)
- Attention type: Axial/factorized attention over height and width; "factorize 2D attention into two 1D attentions along height- and width-axis sequentially." (1 Introduction)
- Computational cost mechanisms: "Axial-attention reduces the complexity to  $\mathcal{O}(hwm)$" and "we adopt local constraints (i.e., a local  $m \times m$  square region as in [65]) in the first few blocks of Full Axial-ResNets, in order to reduce computational cost." (3.2 Axial-Attention)
- Example span settings: "we set the span m to the whole input from the first block, where the feature map is  $56\times 56$" (3.2 Axial-Attention); "Due to the computational cost introduced by the early layers, we set the axial span m=15 in all blocks of Stand-Alone Axial-ResNets." (4.1 ImageNet); "we resort to a large span m=65 in all our axial-attention blocks" for large inputs (3 Method, Axial-DeepLab).

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: Learned relative positional encoding; "a learned relative positional encoding term is incorporated into the affinities" and "We do not consider absolute positional encoding  $q_o^T r_p$ , because they do not generalize well compared to the relative counterpart [65]." (3.1 Position-Sensitive Self-Attention)
- Where it is applied: In attention affinities and values; "we add a key-dependent positional bias term  $k_p^T r_{p-o}^k$ , besides the query-dependent bias  $q_o^T r_{p-o}^q$" and "enable the output  $y_o$  to retrieve relative positions  $r_{p-o}^v$ , besides the content  $v_p$" (3.1 Position-Sensitive Self-Attention).
- Input only / every layer / attention bias: Applied within attention layers as bias terms; "positional encodings are often shared across heads" (3.1 Position-Sensitive Self-Attention).
- Fixed across experiments or modified per task: Not explicitly stated; position-sensitive variants are compared in ablations (see Section 9).

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption: Core variable; "We propose position-sensitive attention layer that makes better use of positional information without adding much computational cost." (1 Introduction)
- Multiple positional encodings compared: Position-sensitive vs previous self-attention; "Position-sensitive attention performs better than previous self-attention [65]" (4.5 Ablation Studies). Absolute positional encoding is explicitly excluded: "We do not consider absolute positional encoding  $q_o^T r_p$ , because they do not generalize well compared to the relative counterpart [65]." (3.1 Position-Sensitive Self-Attention)
- Claim PE choice is not critical or secondary: Not claimed.

## 10. Evidence of Constraint Masking

- Model size/efficiency evidence: "This previous state-of-the-art is attained by our small variant that is 3.8× parameter-efficient and 27× computation-efficient." (Abstract)
- Dataset scale: "We demonstrate the effectiveness of our model on four large-scale datasets." (Abstract)
- Scaling model size: "Increasing the backbone capacity (via large channels) continuously improves the performance." (4.2 COCO)
- Architectural source of gains: "the ability to encode long range context of axial-attention significantly improves the performance on segmentation tasks with large input images." (4.5 Ablation Studies)
- Training tricks: "we strictly follow Panoptic-DeepLab [19], except using a linear warm up Radam [58] Lookahead [92] optimizer" and "We note this change does not improve the results, but smooths our training curves." (4 Experimental Results)

## 11. Architectural Workarounds

- Axial factorization for efficiency: "factorizing 2D self-attention into two 1D selfattentions" to allow attention "within a larger or even global region." (Abstract)
- Local attention windows to reduce cost: "a local  $m \times m$  square region is extracted" for computation reduction (3.1 Position-Sensitive Self-Attention) and "we adopt local constraints (i.e., a local  $m \times m$  square region as in [65]) in the first few blocks of Full Axial-ResNets, in order to reduce computational cost." (3.2 Axial-Attention)
- Span control for memory: "Optionally, one could also use a fixed m value, in order to reduce memory footprint on huge feature maps." (3.2 Axial-Attention); "we resort to a large span m=65 in all our axial-attention blocks" for large inputs (3 Method, Axial-DeepLab).
- Output stride choice: "we extract feature maps with output stride ... 16. We do not pursue output stride 8, since it is computationally expensive." (3 Method, Axial-DeepLab)
- Removing atrous/ASPP modules: "we do not implement the 'atrous' attention module" and "we do not adopt the atrous spatial pyramid pooling module (ASPP) [13,14]" (3 Method, Axial-DeepLab).
- Task-specific heads: "dual decoders, and prediction heads. The heads produce semantic segmentation and class-agnostic instance segmentation, and they are merged by majority voting [89] to form the final panoptic segmentation." (3 Method)

## 12. Explicit Limitations and Non-Claims

- Runtime limitation: "Although our axial-attention model saves M-Adds, it runs slower than convolutional counterparts, as also observed by [65]." (5 Conclusion and Discussion)
- Computational constraint choice: "We do not pursue output stride 8, since it is computationally expensive." (3 Method, Axial-DeepLab)
- Positional encoding exclusion: "We do not consider absolute positional encoding  $q_o^T r_p$ , because they do not generalize well compared to the relative counterpart [65]." (3.1 Position-Sensitive Self-Attention)
- Atrous/ASPP exclusions: "we do not implement the 'atrous' attention module" and "we do not adopt the atrous spatial pyramid pooling module (ASPP) [13,14]" (3 Method, Axial-DeepLab).

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: Multiple datasets within one modality (images: ImageNet, COCO, Mapillary Vistas, Cityscapes).
> – Task structure: Separate evaluations for image classification and for panoptic/instance/semantic segmentation.
> – Representation rigidity: 2D feature maps with axial factorization; output stride fixed to 16; span m set to whole input or fixed values (15/65).
> – Model sharing vs specialization: ImageNet-pretrained Axial-ResNet converted to Axial-DeepLab; shared backbone with separate semantic/instance heads for panoptic.
> – Role of positional encoding: Position-sensitive relative positional encoding is a central architectural choice and ablation variable.

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple tasks: "ImageNet [70] for classification" and "COCO [56], Mapillary Vistas [62], and Cityscapes [22] for panoptic segmentation [45], instance segmentation, and semantic segmentation." (1 Introduction). All evaluations are in image-based vision tasks ("image classification and dense prediction"), and no cross-domain transfer is claimed. (Abstract)
