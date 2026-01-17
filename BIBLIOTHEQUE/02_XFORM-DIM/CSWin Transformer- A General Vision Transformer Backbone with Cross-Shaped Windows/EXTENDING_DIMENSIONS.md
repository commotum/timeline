## 1. Basic Metadata

- Title: "CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows" (Title)
- Authors: "Xiaoyi Dong<sup>1</sup>\*, Jianmin Bao<sup>2</sup>, Dongdong Chen<sup>3</sup>, Weiming Zhang<sup>1</sup>, Nenghai Yu<sup>1</sup>, Lu Yuan<sup>3</sup>, Dong Chen<sup>2</sup>, Baining Guo<sup>2</sup>" (Title)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper frames its contribution as addressing the cost/interaction tradeoff in attention by introducing cross-shaped window self-attention ("A challenging issue in Transformer design is that global self-attention is very expensive to compute whereas local self-attention often limits the field of interactions of each token. To address this issue, we develop the Cross-Shaped Window self-attention mechanism for computing self-attention in the horizontal and vertical stripes in parallel that form a cross-shaped window, with each stripe obtained by splitting the input feature into stripes of equal width." Abstract).

## 3. Tasks Evaluated

- Task name: Image classification
  - Task type: Classification
  - Dataset(s) used: ImageNet-1K
  - Domain: 2D images (vision)
  - Quotes: "we conduct experiments on ImageNet-1K [16] classification" (4. Experiments); "For an input image with size of  $H \times W \times 3$" (3.1. Overall Architecture)

- Task name: Object detection
  - Task type: Detection
  - Dataset(s) used: COCO
  - Domain: 2D images (vision)
  - Quotes: "we conduct experiments on ImageNet-1K [16] classification, COCO [37] object detection, and ADE20K [72] semantic segmentation" (4. Experiments); "Next, we evaluate CSWin Transformer on the COCO objection detection task with the Mask R-CNN [21] and Cascade Mask R-CNN [2] framework respectively." (4.2. COCO Object Detection)

- Task name: Instance segmentation
  - Task type: Segmentation
  - Dataset(s) used: COCO (val2017)
  - Domain: 2D images (vision)
  - Quotes: "Object detection and instance segmentation performance on the COCO val2017 with the Mask R-CNN framework." (Table 4 caption); "COCO Object Detection and Instance Segmentation. We use two classical object detection frameworks: Mask R-CNN [21] and Cascade Mask R-CNN [2]" (Experiment Details)

- Task name: Semantic segmentation
  - Task type: Segmentation
  - Dataset(s) used: ADE20K
  - Domain: 2D images (vision)
  - Quotes: "ADE20K semantic segmentation task" (Abstract); "We further investigate the capability of CSWin Transformer for Semantic Segmentation on the ADE20K [72] dataset." (4.3. ADE20K Semantic Segmentation)

## 4. Domain and Modality Scope

- Single domain, multiple domains within the same modality, or multiple modalities? The evaluation is within a single modality (vision images), across multiple datasets for vision tasks: "general-purpose vision tasks" (Abstract) and "For an input image with size of  $H \times W \times 3$" (3.1. Overall Architecture).
- Does the paper claim domain generalization or cross-domain transfer? Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Image classification | Not specified across tasks; single-task training described | Yes (for higher-resolution reporting) | Not specified | "all our models are trained for 300 epochs with the input size of  $224 \times 224$" and "When reporting the results of  $384 \times 384$  input, we fine-tune the models for 30 epochs" (4.1. ImageNet-1K Classification) |
| Object detection | Yes (backbone pretraining on ImageNet-1K) | Yes | Yes (Mask R-CNN / Cascade Mask R-CNN) | "we pretrain the backbones on the ImageNet-1K dataset and follow the finetuning strategy used in Swin Transformer [38] on the COCO training set" and "with the Mask R-CNN [21] and Cascade Mask R-CNN [2] framework respectively" (4.2. COCO Object Detection); "For Mask R-CNN, we train it with ImageNet-1K pretrained model with two settings:  $1 \times$  schedule and  $3 \times$ +MS schedule." (Experiment Details) |
| Instance segmentation | Yes (backbone pretraining on ImageNet-1K) | Yes | Yes (Mask R-CNN / Cascade Mask R-CNN) | "Object detection and instance segmentation performance on the COCO val2017 with the Mask R-CNN framework. The FLOPs (G) are measured at resolution  $800 \times 1280$ , and the models are pre-trained on the ImageNet-1K." (Table 4 caption); "For Mask R-CNN, we train it with ImageNet-1K pretrained model with two settings:  $1 \times$  schedule and  $3 \times$ +MS schedule." (Experiment Details) |
| Semantic segmentation | Not fully specified; ImageNet-21K pretraining is stated for some models | Yes (for ImageNet-21K pretrained models) | Yes (Semantic FPN / Upernet) | "Here we employ the semantic FPN [33] and Upernet [61] as the basic framework." and "† means the model is pretrained on ImageNet-21K and finetuned with  $640 \times 640$  resolution." (4.3. ADE20K Semantic Segmentation; Table 6 caption) |

## 6. Input and Representation Constraints

- Input modality and dimensionality: "For an input image with size of  $H \times W \times 3$" (3.1. Overall Architecture).
- Tokenization / patching: "leverage the overlapped convolutional token embedding ( $7 \times 7$  convolution layer with stride 4)) to obtain  $\frac{H}{4} \times \frac{W}{4}$  patch tokens" (3.1. Overall Architecture).
- Hierarchical token downsampling: "A convolution layer  $(3 \times 3$ , stride 2) is used between two adjacent stages to reduce the number of tokens and double the channel dimension. Therefore, the constructed feature maps have  $\frac{H}{2^{i+1}} \times \frac{W}{2^{i+1}}$  tokens for the  $i^{th}$  stage" (3.1. Overall Architecture).
- Variable vs fixed input resolution: "LePE naturally supports arbitrary input resolutions" (Abstract), but training/evaluation uses fixed sizes such as "input size of  $224 \times 224$" and " $384 \times 384$  input" (4.1. ImageNet-1K Classification).
- Stripe-width divisibility constraint: "To make the intermediate feature map size divisible by sw for  $224 \times 224$  input, we empirically set sw to 1, 2, 7, 7 for four stages by default." (3.2. Cross-Shaped Window Self-Attention)
- Positional encoding locality constraint: "we set a distance threshold to the LePE and set it to 0 if the Chebyshev distance of token i and j is greater than a threshold  $\tau$  ( $\tau=3$  in the default setting)." (3.2. Cross-Shaped Window Self-Attention)
- Detection input resizing constraints: "For  $1 \times$  schedule, we train the model with single-scale input (image is resized to the shorter side of 800 pixels, while the longer side does not exceed 1333 pixels)" and "For 3×+MS schedule, we train the model with multi-scale input (image is resized to the shorter side between 480 and 800 while the longer side is no longer than 1333)" (Experiment Details).
- Segmentation training input size: "All the models are trained with input size  $512 \times 512$ ." (Experiment Details)

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Fixed or variable sequence length: Variable with input size, e.g., "For an input image with size of  $H \times W \times 3$" and tokens scale with resolution (3.1. Overall Architecture).
- Attention type: Windowed + hierarchical (cross-shaped window stripes and staged hierarchy). Evidence: "Cross-Shaped Window self-attention mechanism for computing self-attention in the horizontal and vertical stripes in parallel that form a cross-shaped window" (Abstract); "the whole network consists of four stages" (3.1. Overall Architecture).
- Mechanisms to manage computational cost: "we adjust the stripe width according to the depth of the network: small widths for shallow layers and larger widths for deep layers" (1. Introduction); "A convolution layer  $(3 \times 3$ , stride 2) is used between two adjacent stages to reduce the number of tokens" (3.1. Overall Architecture).

## 8. Positional Encoding (Critical Section)

- Mechanism: Locally-enhanced Positional Encoding (LePE), implemented as a per-channel bias on values within each block. Evidence: "we introduce an effective positional encoding, *Locally-enhanced Positional Encoding* (LePE)" (1. Introduction); "our LePE operates directly upon V and acts as a parallel module" (Figure 3 caption); "our LePE is a per-channel bias" (3.2. Cross-Shaped Window Self-Attention).
- Where applied: Within each Transformer block, parallel to attention results: "our LePE imposes the positional information within each Transformer block and directly operates on the attention results instead of the attention calculation" and "LePE is added as a parallel module to the self-attention branch." (1. Introduction; 3.1. Overall Architecture)
- Fixed vs compared: It is the default choice but also compared against alternatives: "we compare our LePE with other recent positional encoding mechanisms(APE [17], CPE [12], and RPE [46])" (4.5. Ablation Study: Positional Encoding Comparison).

## 9. Positional Encoding as a Variable

- Core variable or fixed assumption? Both: LePE is part of the core architecture, but positional encoding is explicitly varied in ablations: "we introduce an effective positional encoding, *Locally-enhanced Positional Encoding* (LePE)" (1. Introduction) and "we compare our LePE with other recent positional encoding mechanisms(APE [17], CPE [12], and RPE [46])" (4.5. Ablation Study: Positional Encoding Comparison).
- Are multiple positional encodings compared? Yes: "we compare our LePE with other recent positional encoding mechanisms(APE [17], CPE [12], and RPE [46])" (4.5. Ablation Study).
- Does the paper claim PE choice is "not critical" or secondary? Not stated.

## 10. Evidence of Constraint Masking

- Model sizes: "CSWin-T | 64   | 1,2,21,1 | 1,2,7,7 | 2,4,8,16   | 23M     | 4.3G  |" and "CSWin-L | 144  | 2,4,32,2 | 1,2,7,7 | 6,12,24,48 | 173M    | 31.5G |" (Table 1).
- Dataset sizes: "ImageNet-21K dataset, which contains 14.2M images and 21K classes." (4.1. ImageNet-1K Classification)
- Performance gains attributed to scaling data: "By further pretraining on the larger dataset ImageNet-21K, we achieve 87.5% Top-1 accuracy on ImageNet-1K" (Abstract); "the large-scale data of ImageNet-21K brings a 1.6%~1.7% gain" (4.1. ImageNet-1K Classification).
- Performance gains attributed to architectural hierarchy/attention design: "we develop the Cross-Shaped Window self-attention mechanism" and "Incorporated with these designs and a hierarchical structure, CSWin Transformer demonstrates competitive performance on common vision tasks." (Abstract)
- Training tricks mentioned: "We apply increasing stochastic depth [29] augmentation for CSWin-T, CSWin-S, and CSWin-B" (4.1. ImageNet-1K Classification).

## 11. Architectural Workarounds

- Cross-shaped window attention to limit cost while expanding receptive field: "Cross-Shaped Window self-attention mechanism for computing self-attention in the horizontal and vertical stripes in parallel that form a cross-shaped window" (Abstract).
- Parallel multi-head grouping: "We split the multi-heads into **parallel** groups and apply different self-attention operations onto different groups." (1. Introduction)
- Stripe width scheduling: "we adjust the stripe width according to the depth of the network: small widths for shallow layers and larger widths for deep layers." (1. Introduction)
- Hierarchical stages with token reduction: "the whole network consists of four stages. A convolution layer  $(3 \times 3$ , stride 2) is used between two adjacent stages to reduce the number of tokens and double the channel dimension." (3.1. Overall Architecture)
- Overlapped convolutional token embedding: "overlapped convolutional token embedding ( $7 \times 7$  convolution layer with stride 4))" (3.1. Overall Architecture)
- LePE as a parallel branch to attention: "LePE is added as a parallel module to the self-attention branch." (3.1. Overall Architecture)

## 12. Explicit Limitations and Non-Claims

- Limitations: Not specified.
- Future work statement: "We are looking forward to applying it for more vision tasks." (5. Conclusion)
- Explicit non-claims about unrestrained multi-task learning or cross-domain learning: Not specified.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: Single modality (vision images), multiple datasets for vision tasks; no cross-domain transfer claim.
> – Task structure: Separate evaluations for classification, object detection/instance segmentation, and semantic segmentation.
> – Representation rigidity: 2D image inputs with fixed tokenization (overlapped conv, staged downsampling), variable resolution supported but many experiments fixed-size.
> – Model sharing vs specialization: Backbone pretraining with task-specific fine-tuning and task frameworks (Mask R-CNN, Semantic FPN/Upernet).
> – Role of positional encoding: LePE is default per-block bias, but PE is explicitly varied in ablation.

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple tasks within vision using the same image modality ("ImageNet-1K [16] classification, COCO [37] object detection, and ADE20K [72] semantic segmentation"; 4. Experiments; and "For an input image with size of  $H \times W \times 3$"; 3.1. Overall Architecture). There is no claim of cross-domain transfer or multiple modalities; the framing is as a "general-purpose vision" backbone rather than unrestrained multi-domain learning (Abstract).
