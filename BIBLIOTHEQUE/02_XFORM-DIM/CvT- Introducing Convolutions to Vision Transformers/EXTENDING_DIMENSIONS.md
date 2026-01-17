## 1. Basic Metadata

- Title: "CvT: Introducing Convolutions to Vision Transformers" (Title/header)
- Authors: "Haiping  $Wu^{1,2*}$  Bin  $Xiao^{2\dagger}$  Noel Codella<sup>2</sup> Mengchen Liu<sup>2</sup> Xiyang Dai<sup>2</sup> Lu Yuan<sup>2</sup> Lei Zhang<sup>2</sup>" (Title/header)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

"We present in this paper a new architecture, named Convolutional vision Transformer (CvT), that improves Vision Transformer (ViT) in performance and efficiency by introducing convolutions into ViT to yield the best of both designs." (Abstract)

## 3. Tasks Evaluated

- Task name: Image classification (ImageNet-1k)
  - Task type: Classification
  - Dataset(s) used: ImageNet-1k
  - Domain: Natural images
  - Quotes: "We validate CvT by conducting extensive experiments, showing that this approach achieves state-of-the-art performance over other Vision Transformers and ResNets on ImageNet-1k, with fewer parameters and lower FLOPs." (Abstract); "For evaluation, we use the ImageNet dataset, with 1.3M images and 1k classes" (Section 4.1. Setup)

- Task name: Image classification (ImageNet Real)
  - Task type: Classification
  - Dataset(s) used: ImageNet Real
  - Domain: Natural images
  - Quotes: "We compare our method with state-of-the-art classification methods including Transformer-based models and representative CNN-based models on ImageNet [9], ImageNet Real [2] and ImageNet V2 [26] datasets in Table 3." (Section 4.2. Comparison to state of the art); "Table 3: Accuracy of manual designed architecture on ImageNet [9], ImageNet Real [2] and ImageNet V2 matched frequency [26]." (Table 3)

- Task name: Image classification (ImageNet V2 matched frequency)
  - Task type: Classification
  - Dataset(s) used: ImageNet V2 matched frequency
  - Domain: Natural images
  - Quotes: "We compare our method with state-of-the-art classification methods including Transformer-based models and representative CNN-based models on ImageNet [9], ImageNet Real [2] and ImageNet V2 [26] datasets in Table 3." (Section 4.2. Comparison to state of the art); "Table 3: Accuracy of manual designed architecture on ImageNet [9], ImageNet Real [2] and ImageNet V2 matched frequency [26]." (Table 3)

- Task name: Image classification (CIFAR-10)
  - Task type: Classification
  - Dataset(s) used: CIFAR-10
  - Domain: Natural images
  - Quotes: "We further transfer the models pretrained on ImageNet-22k to downstream tasks, including CIFAR-10/100 [19], Oxford-IIIT-Pet [23], Oxford-IIIT-Flower [22], following [18, 11]." (Section 4.1. Setup); "| Model         | Param<br>(M) | CIFAR<br>10  | CIFAR<br>100 | Pets         | Flowers<br>102 |" (Table 4)

- Task name: Image classification (CIFAR-100)
  - Task type: Classification
  - Dataset(s) used: CIFAR-100
  - Domain: Natural images
  - Quotes: "We further transfer the models pretrained on ImageNet-22k to downstream tasks, including CIFAR-10/100 [19], Oxford-IIIT-Pet [23], Oxford-IIIT-Flower [22], following [18, 11]." (Section 4.1. Setup); "| Model         | Param<br>(M) | CIFAR<br>10  | CIFAR<br>100 | Pets         | Flowers<br>102 |" (Table 4)

- Task name: Image classification (Oxford-IIIT Pets)
  - Task type: Classification
  - Dataset(s) used: Oxford-IIIT Pets
  - Domain: Natural images
  - Quotes: "We further transfer the models pretrained on ImageNet-22k to downstream tasks, including CIFAR-10/100 [19], Oxford-IIIT-Pet [23], Oxford-IIIT-Flower [22], following [18, 11]." (Section 4.1. Setup); "| Model         | Param<br>(M) | CIFAR<br>10  | CIFAR<br>100 | Pets         | Flowers<br>102 |" (Table 4)

- Task name: Image classification (Oxford-IIIT Flowers-102)
  - Task type: Classification
  - Dataset(s) used: Oxford-IIIT Flowers-102
  - Domain: Natural images
  - Quotes: "We further transfer the models pretrained on ImageNet-22k to downstream tasks, including CIFAR-10/100 [19], Oxford-IIIT-Pet [23], Oxford-IIIT-Flower [22], following [18, 11]." (Section 4.1. Setup); "| Model         | Param<br>(M) | CIFAR<br>10  | CIFAR<br>100 | Pets         | Flowers<br>102 |" (Table 4)

## 4. Domain and Modality Scope

- Evaluation is performed on multiple datasets within the same modality (images) and same overall domain (natural image classification): "In this section, we evaluate the CvT model on large-scale image classification datasets and transfer to various down-stream datasets." (Section 4. Experiments); "We further transfer the models pretrained on ImageNet-22k to downstream tasks, including CIFAR-10/100 [19], Oxford-IIIT-Pet [23], Oxford-IIIT-Flower [22]" (Section 4.1. Setup).
- Multiple modalities? Not indicated; all evaluations are image datasets (see quotes above).
- Domain generalization or cross-domain transfer: Not claimed. The paper only claims transfer within vision tasks: "We further investigate the ability of our models to transfer by fine-tuning models on various tasks, with all models being pre-trained on ImageNet-22k." (Section 4.3. Downstream task transfer)

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| ImageNet-1k classification | Not specified (training on ImageNet-1k and/or pretrain+finetune reported) | Yes (in pretrain+finetune setting) | Not specified | "Subscript  $_{22k}$  indicates the model pre-trained on ImageNet22k [9], and finetuned on ImageNet1k" (Table 3); "As in ViT [30], we pre-train our models at resolution  $224 \times 224$ , and fine-tune at resolution of  $384 \times 384$ ." (Section 4.1. Setup) |
| ImageNet Real classification | Not specified | Not specified beyond ImageNet-1k fine-tuning | Not specified | "We compare our method with state-of-the-art classification methods including Transformer-based models and representative CNN-based models on ImageNet [9], ImageNet Real [2] and ImageNet V2 [26] datasets in Table 3." (Section 4.2. Comparison to state of the art); "Subscript  $_{22k}$  indicates the model pre-trained on ImageNet22k [9], and finetuned on ImageNet1k" (Table 3) |
| ImageNet V2 matched frequency classification | Not specified | Not specified beyond ImageNet-1k fine-tuning | Not specified | "We compare our method with state-of-the-art classification methods including Transformer-based models and representative CNN-based models on ImageNet [9], ImageNet Real [2] and ImageNet V2 [26] datasets in Table 3." (Section 4.2. Comparison to state of the art); "Subscript  $_{22k}$  indicates the model pre-trained on ImageNet22k [9], and finetuned on ImageNet1k" (Table 3) |
| CIFAR-10 classification | No (fine-tuned per task) | Yes | Not specified | "We fine-tune each model with a total batch size of 512, for 20,000 steps on ImageNet-1k, 10,000 steps on CIFAR-10 and CIFAR-100, and 500 steps on Oxford-IIIT Pets and Oxford-IIIT Flowers-102." (Section 4.1. Setup) |
| CIFAR-100 classification | No (fine-tuned per task) | Yes | Not specified | "We fine-tune each model with a total batch size of 512, for 20,000 steps on ImageNet-1k, 10,000 steps on CIFAR-10 and CIFAR-100, and 500 steps on Oxford-IIIT Pets and Oxford-IIIT Flowers-102." (Section 4.1. Setup) |
| Oxford-IIIT Pets classification | No (fine-tuned per task) | Yes | Not specified | "We fine-tune each model with a total batch size of 512, for 20,000 steps on ImageNet-1k, 10,000 steps on CIFAR-10 and CIFAR-100, and 500 steps on Oxford-IIIT Pets and Oxford-IIIT Flowers-102." (Section 4.1. Setup) |
| Oxford-IIIT Flowers-102 classification | No (fine-tuned per task) | Yes | Not specified | "We fine-tune each model with a total batch size of 512, for 20,000 steps on ImageNet-1k, 10,000 steps on CIFAR-10 and CIFAR-100, and 500 steps on Oxford-IIIT Pets and Oxford-IIIT Flowers-102." (Section 4.1. Setup) |

## 6. Input and Representation Constraints

- Fixed input sizes used in experiments: "Input image size is  $224 \times 224$  by default." (Table 2); "Unless otherwise stated, all ImageNet models are trained with an  $224 \times 224$  input size." (Section 4.1. Setup); "we pre-train our models at resolution  $224 \times 224$ , and fine-tune at resolution of  $384 \times 384$ ." (Section 4.1. Setup)
- Variable-resolution capability claimed due to removing positional embedding: "makes it readily capable of accommodating variable resolutions of input images" (Section 1. Introduction); "CvT is able to completely remove the positional embedding, providing the possibility of simplifying adaption to more vision tasks without requiring a re-designing of the embedding." (Section 4.4. Ablation Study)
- 2D input assumption and fixed 2D token map: "given a 2D image or a 2D-reshaped output to-ken map from a previous stage  $x_{i-1} \in \mathbb{R}^{H_{i-1} \times W_{i-1} \times C_{i-1}}$" (Section 3.1. Convolutional Token Embedding)
- Tokenization via convolution with kernel/stride/padding: "$f(\cdot)$  is 2D convolution operation of kernel size  $s \times s$ , stride s-o and p padding (to deal with boundary conditions)." (Section 3.1. Convolutional Token Embedding)
- Sequence length derived from spatial size: "$f(x_{i-1})$  is then flattened into size  $H_iW_i \times C_i$" (Section 3.1. Convolutional Token Embedding)
- Overlapping vs non-overlapping patches: "a convolutional token embedding that performs an overlapping convolution operation with stride on a 2D-reshaped token map" (Section 3. Convolutional vision Transformer); "images are split into discrete non-overlapping patches ( $e.g.\ 16 \times 16$ )." (Section 1. Introduction)

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Sequence length fixed or variable: Variable with spatial size: "$f(x_{i-1})$  is then flattened into size  $H_iW_i \times C_i$" (Section 3.1. Convolutional Token Embedding); variable input resolutions are discussed: "variable resolutions of input images" (Section 1. Introduction).
- Attention type: Global, with hierarchical stages: "input into repeated standard Transformer layers to model global relations for classification." (Section 1. Introduction); "a multi-stage hierarchy design borrowed from CNNs" (Section 3. Convolutional vision Transformer).
- Mechanisms to manage computational cost: "the stride of convolution can be used to subsample the key and value matrices to improve efficiency by  $4\times$  or more" (Section 3. Convolutional vision Transformer); "progressively decrease the sequence length while simultaneously increasing the dimension of token features across stages" (Section 3. Convolutional vision Transformer).

## 8. Positional Encoding (Critical Section)

- Mechanism used: None/implicit in CvT by default: "we do not sum the ad-hod position embedding to the tokens." (Section 3. Convolutional vision Transformer); "position embeddings have been removed from CvT by default." (Section 4.4. Ablation Study)
- Where applied: Ablated in different stages but default is none: "| d      | CvT-13 | 20        | First stage | 81.4               |" (Table 5); "| e      | CvT-13 | 20        | Last stage  | 81.4               |" (Table 5); "| f      | CvT-13 | 20        | N/A         | 81.6               |" (Table 5)
- Fixed across experiments vs modified: Positional embedding is ablated/compared: "The results are shown in Table 5, and demonstrate that removing position embedding of our model does not degrade the performance." (Section 4.4. Ablation Study)

## 9. Positional Encoding as a Variable

- Treated as a core research variable (ablation): "we study whether position embedding is still needed for CvT. The results are shown in Table 5" (Section 4.4. Ablation Study)
- Multiple positional encodings compared: "| d      | CvT-13 | 20        | First stage | 81.4               |" (Table 5); "| e      | CvT-13 | 20        | Last stage  | 81.4               |" (Table 5); "| f      | CvT-13 | 20        | N/A         | 81.6               |" (Table 5)
- Claim that PE is not critical/secondary: "removing position embedding of our model does not degrade the performance. Therefore, position embeddings have been removed from CvT by default." (Section 4.4. Ablation Study)

## 10. Evidence of Constraint Masking

- Model sizes and compute: "CvT-13 and CvT-21 as basic models, with 19.98M and 31.54M parameters." (Section 4.1. Setup); "CvT-W24 (W stands for Wide), resulting 298.3M parameters" (Section 4.1. Setup); "Our smallest model CvT-13 with 20M parameters and 4.5G FLOPs" (Section 4.2. Comparison to state of the art)
- Dataset sizes and scaling data: "ImageNet dataset, with 1.3M images and 1k classes, as well as its superset ImageNet-22k with 22k classes and 14M images" (Section 4.1. Setup); "performance gains are maintained when pretrained on larger datasets (e.g. ImageNet-22k)" (Abstract)
- Performance gains attributed to architectural hierarchy/convolutions: "Extensive experiments demonstrate that the introduced convolutional token embedding and convolutional projection, along with the multi-stage design of the network enabled by convolutions, make our CvT architecture achieve superior performance while maintaining computational efficiency." (Section 5. Conclusion)
- Evidence of scaling model/data for gains: "Furthermore, when more data are involved, our wide model CvT-W24\\* pretrained on ImageNet-22k reaches to **87.7**% Top-1 Accuracy on ImageNet *without extra data*" (Section 4.2. Comparison to state of the art)

## 11. Architectural Workarounds

- Hierarchical stages with downsampling: "a multi-stage hierarchy design borrowed from CNNs" (Section 3. Convolutional vision Transformer); "three stages in total are used in this work" (Section 3. Convolutional vision Transformer)
- Convolutional token embedding with overlapping patches and stride: "a convolutional token embedding that performs an overlapping convolution operation with stride on a 2D-reshaped token map" (Section 3. Convolutional vision Transformer)
- Convolutional projection for attention: "the linear projection prior to every self-attention block in the Transformer module is replaced with our proposed convolutional projection" (Section 3. Convolutional vision Transformer)
- K/V subsampling for efficiency: "the stride of convolution can be used to subsample the key and value matrices to improve efficiency by  $4\times$  or more" (Section 3. Convolutional vision Transformer)
- Classification token only at final stage: "the classification token is added only in the last stage." (Section 3. Convolutional vision Transformer)
- Removal of positional embedding: "position embeddings have been removed from CvT by default." (Section 4.4. Ablation Study)

## 12. Explicit Limitations and Non-Claims

- Limitations: Not specified.
- Non-claims (e.g., open-world learning, unrestrained multi-task learning, meta-learning): Not specified.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: single modality (vision) across multiple natural-image datasets (ImageNet/CIFAR/Pets/Flowers).
> - Task structure: multiple classification tasks; no detection/segmentation evaluated.
> - Representation rigidity: 2D image inputs with convolutional tokenization and fixed training resolutions (224/384), though variable resolution is claimed via no PE.
> - Model sharing vs specialization: pretrain on ImageNet-22k and fine-tune per downstream dataset; no joint multi-task training reported.
> - Role of positional encoding: explicitly ablated and removed by default.

### 14. Final Classification

**Multi-task, single-domain**

The paper evaluates multiple classification tasks on different natural-image datasets (ImageNet, CIFAR-10/100, Oxford-IIIT Pets/Flowers), all within the same image modality: "In this section, we evaluate the CvT model on large-scale image classification datasets and transfer to various down-stream datasets." (Section 4. Experiments) and "We further transfer the models pretrained on ImageNet-22k to downstream tasks, including CIFAR-10/100 [19], Oxford-IIIT-Pet [23], Oxford-IIIT-Flower [22], following [18, 11]." (Section 4.1. Setup). It reports fine-tuning per dataset rather than joint multi-task training, indicating multiple tasks within a single domain rather than unrestrained multi-domain learning (Section 4.1. Setup).
