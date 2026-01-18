## 1. Basic Metadata

Title: "Rethinking and Improving Relative Position Encoding for Vision Transformer" (paper header)
Authors: "Kan Wu<sup>1,2,\*</sup>, Houwen Peng<sup>2,\*,†</sup>, Minghao Chen<sup>2</sup>, Jianlong Fu<sup>2</sup>, Hongyang Chao<sup>1</sup>" (paper header)
Year: Year not specified.
Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

Abstract: The paper notes that in vision transformers "its efficacy is not well studied and even remains controversial, e.g., whether relative position encoding can work equally well as absolute position?" and proposes "new relative position encoding methods dedicated to 2D images, called image RPE (iRPE)."

## 3. Tasks Evaluated

Task name: Image classification
Task type: Classification
Dataset(s) used: "ImageNet [4]"
Domain: Domain not explicitly stated; dataset named "ImageNet [4]".
Quotes: "We compare our proposed methods with the state-of-the-art methods on image classification tasks." (Section 4.3. Comparison on Image Classification); "Table 1: Ablation of our relative position encoding methods on ImageNet [4]." (Section 4.1. Implementation Details)

Task name: Fine-grained image classification / transfer learning
Task type: Classification
Dataset(s) used: "Stanford Cars" and "CUB200_2011"
Domain: Domain not explicitly stated; datasets named "Stanford Cars" and "CUB200_2011".
Quotes: "We finetune the pretrained models on Stanford Cars and CUB200_2011 datasets using the resolution 224x224 and 300 epochs." (Section 7. Transfer Learning on Fine-grained Datasets)

Task name: Object detection
Task type: Detection
Dataset(s) used: "COCO 2017 detection dataset [12]"
Domain: Domain not explicitly stated; dataset named "COCO 2017 detection dataset [12]".
Quotes: "Then, we compare the proposed methods with the state-of-the-art methods on image classification and object detection tasks." (Section 4. Experiments); "We further evaluate it on COCO 2017 detection dataset [12]." (Section 4.4. Comparison on Object Detection)

## 4. Domain and Modality Scope

Single domain?: Not explicitly stated; evaluation uses multiple image datasets: "ImageNet [4]" (Section 4.1. Implementation Details), "COCO 2017 detection dataset [12]" (Section 4.4. Comparison on Object Detection), and "Stanford Cars and CUB200_2011 datasets" (Section 7. Transfer Learning on Fine-grained Datasets).
Multiple domains within the same modality?: Not explicitly stated; tasks are "image classification and object detection tasks." (Section 4. Experiments)
Multiple modalities?: Not claimed; only image tasks are described ("image classification and object detection tasks," Section 4. Experiments).
Domain generalization / cross-domain transfer?: Transfer learning is reported ("We finetune the pretrained models on Stanford Cars and CUB200_2011 datasets..." Section 7), but domain generalization or cross-domain transfer is not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Image classification (ImageNet) | Not stated; separate baselines are used for classification vs detection | No; "All models are trained from scratch for 300 epochs with 8 NVIDIA Tesla V100 GPUs." | Not stated; "An extra trainable classification token is added into the sequence for classification." | "We select DeiT [22] as the baseline." (Section 4.3. Comparison on Image Classification); "We use the transformer-based detection model DETR [1] as our baseline." (Section 4.4. Comparison on Object Detection) |
| Fine-grained image classification / transfer learning | Not stated; pretrained models are fine-tuned per dataset | Yes; "We finetune the pretrained models on Stanford Cars and CUB200_2011 datasets..." | Not stated; "An extra trainable classification token is added into the sequence for classification." | "We finetune the pretrained models on Stanford Cars and CUB200_2011 datasets using the resolution 224x224 and 300 epochs." (Section 7. Transfer Learning on Fine-grained Datasets) |
| Object detection (COCO) | Not stated; separate baseline for detection | Not stated; backbone is pretrained | Not stated; "The transformer outputs a certain number of bounding boxes." | "We use the transformer-based detection model DETR [1] as our baseline." (Section 4.4. Comparison on Object Detection); "The backbone model of DETR [1] is ResNet-50 [10], pretrained on ImageNet [4], and the BatchNorm layers are frozen during training." (Section 5. Training and Test Settings of DETR) |

## 6. Input and Representation Constraints

Fixed or variable input resolution: "For training, the images are split into 14x14 non-overlapping patches." (Section 4.1. Implementation Details); "Object detection uses a much higher resolution input compared to classification, leading to a much longer input sequence." (Section 4.2. Analysis on Relative Position Encoding); "We finetune the pretrained models on Stanford Cars and CUB200_2011 datasets using the resolution 224x224 and 300 epochs." (Section 7. Transfer Learning on Fine-grained Datasets)
Fixed patch size: "In ViT [6] and DeiT [22] models, an image is split into multiple fixed-size patches." (Section 5. Related Work)
Fixed number of tokens: "For training, the images are split into 14x14 non-overlapping patches." (Section 4.1. Implementation Details); "An extra trainable classification token is added into the sequence for classification." (Section 5. Related Work); "The number of queries is 100." (Section 5. Training and Test Settings of DETR)
Fixed dimensionality (2D): "the inputs are usually 2D images or video sequences" (Section 1. Introduction); "new relative position encoding methods dedicated to 2D images, called image RPE (iRPE)." (Abstract)
Backbone representation (DETR): "In DETR [1], a CNN backbone is used for feature extraction first. It outputs a feature map downsampled  $32\times$ . Then it is flatten and fed to a transformer." (Section 5. Related Work)
Padding or resizing requirements: "The image is cropped such that the shortest side is at least 480 and at most 800 pixels while the longest at most 1333." (Section 5. Training and Test Settings of DETR)

## 7. Context Window and Attention Structure

Maximum sequence length: Not specified.
Sequence length fixed or variable: "For training, the images are split into 14x14 non-overlapping patches." (Section 4.1. Implementation Details); "Object detection uses a much higher resolution input compared to classification, leading to a much longer input sequence." (Section 4.2. Analysis on Relative Position Encoding); "The number of queries is 100." (Section 5. Training and Test Settings of DETR)
Attention type: "The shallow layers in transformer are also global attentions, which pay attention to the whole image (consisting of small patches)." (Section 4.5. Visualization)
Mechanisms to manage computational cost: "We introduce an efficient implementation of relative encoding, which reduces the computational cost from the original  $\mathcal{O}(n^2d)$  to  $\mathcal{O}(nkd)$ , where  $k \ll n$ ." (Section 1. Introduction); "Such index function can largely reduce computation costs and the number of parameters for long sequence (e.g., high resolution images)." (Section 3.2. Proposed Relative Position Encoding Methods)

## 8. Positional Encoding (Critical Section)

Positional encoding mechanism used: "The original self-attention considers the *absolute position* [23], and add the absolute positional encodings  $\mathbf{p} = (\mathbf{p}_1, \dots, \mathbf{p}_n)$  to the input token embedding  $\mathbf{x}$" (Section 2.2. Position Encoding); "Relative position encoding (RPE) is commonly calculated via a look-up table with learnable parameters interacting with queries and keys in self-attention modules [18]." (Section 1. Introduction); "We then propose new relative position encoding methods dedicated to 2D images, called image RPE (iRPE)." (Abstract)
Where it is applied: "The encoding vectors are embedded into the self-attention module" (Section 2.2. Position Encoding); "The relative position encoding is added into all self-attention layers. If not specified, the relative position encoding is only added on keys." (Section 4.1. Implementation Details); "A learnable or sinusoid absolute position encoding is added in both transformer encoder and decoder." (Section 5. Related Work)
Fixed vs modified/ablated: "In this section, we first provide some analysis by comparing different position embeddings" (Section 4. Experiments); "We empirically demonstrate that relative position encoding can replace the absolute encoding for image classification task. Meanwhile, the absolute encoding is necessary for object detection, where the pixel position is important for object localization." (Section 1. Introduction)

## 9. Positional Encoding as a Variable

Core research variable?: "we first review existing relative position encoding methods and analyze their pros and cons when applied in vision transformers. We then propose new relative position encoding methods dedicated to 2D images" (Abstract)
Multiple positional encodings compared?: "In this section, we first provide some analysis by comparing different position embeddings" (Section 4. Experiments); "Table 4: Component-wise analysis on ImageNet [4]. We add contextual product shared-head relative position encodings into DeiT-S [22]. The number of buckets is 50. Abs Pos. represents the absolute position encoding." (Section 4.2. Analysis on Relative Position Encoding)
PE claimed "not critical" or secondary?: Not claimed.

## 10. Evidence of Constraint Masking

Model size(s): "DeiT-S [22]                | 22M        | $224^{2}$ | 4613          | 79.9" (Table 5: Comparison on ImageNet [4]); "DeiT-B [22]                | 86M        | $224^{2}$ | 17592         | 81.8" (Table 5: Comparison on ImageNet [4])
Dataset size(s): Not specified.
Attribution of gains (scaling vs architecture/training): "Experiments demonstrate that solely due to the proposed encoding methods, DeiT [22] and DETR [1] obtain up to 1.5% (top-1 Acc) and 1.3% (mAP) stable improvements over their original versions on ImageNet and COCO respectively, without tuning any extra hyperparameters such as learning rate and weight decay." (Abstract)

## 11. Architectural Workarounds

- "We introduce an efficient implementation of relative encoding, which reduces the computational cost from the original  $\mathcal{O}(n^2d)$  to  $\mathcal{O}(nkd)$ , where  $k \ll n$ ." (Section 1. Introduction)
- "Such index function can largely reduce computation costs and the number of parameters for long sequence (e.g., high resolution images)." (Section 3.2. Proposed Relative Position Encoding Methods)
- "In DETR [1], a CNN backbone is used for feature extraction first. It outputs a feature map downsampled  $32\times$ . Then it is flatten and fed to a transformer." (Section 5. Related Work)
- "In ViT [6] and DeiT [22] models, an image is split into multiple fixed-size patches." (Section 5. Related Work)
- "The number of queries is 100." (Section 5. Training and Test Settings of DETR)

## 12. Explicit Limitations and Non-Claims

- "We empirically demonstrate that relative position encoding can replace the absolute encoding for image classification task. Meanwhile, the absolute encoding is necessary for object detection, where the pixel position is important for object localization." (Section 1. Introduction)
- "In future work, we plan to extend our method to other attention-based models and scenarios, such as high-resolution input tasks like semantic segmentation [30], and non-pixel input tasks like point cloud classification [29, 9]." (Section 6. Conclusions and Remarks)
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not specified.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Image-only evaluations across "image classification and object detection tasks" and fine-grained datasets like "Stanford Cars and CUB200_2011 datasets." (Section 4. Experiments; Section 7. Transfer Learning on Fine-grained Datasets)
> - Task structure: Separate task-specific baselines are used ("We select DeiT [22] as the baseline." and "We use the transformer-based detection model DETR [1] as our baseline.") (Sections 4.3 and 4.4)
> - Representation rigidity: Fixed patching and resolutions are specified ("14x14 non-overlapping patches" and "resolution 224x224"), plus detection-specific cropping. (Section 4.1; Section 7; Section 5 Training and Test Settings of DETR)
> - Model sharing vs specialization: Pretrained/fine-tuned pipelines are used for some tasks ("The backbone model of DETR [1] is ResNet-50 [10], pretrained on ImageNet [4]" and "We finetune the pretrained models on Stanford Cars and CUB200_2011 datasets..."). (Section 5 Training and Test Settings of DETR; Section 7)
> - Role of positional encoding: Central experimental variable with task-dependent conclusions ("relative position encoding can replace the absolute encoding for image classification task" but "the absolute encoding is necessary for object detection"). (Section 1. Introduction)

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple tasks, explicitly "image classification and object detection tasks," and also reports fine-tuning on "Stanford Cars and CUB200_2011 datasets" (Section 4. Experiments; Section 7. Transfer Learning on Fine-grained Datasets). All evaluations are within the image modality, and separate baselines are used ("We select DeiT [22] as the baseline." and "We use the transformer-based detection model DETR [1] as our baseline."), with no cross-domain or multi-modality claim.
