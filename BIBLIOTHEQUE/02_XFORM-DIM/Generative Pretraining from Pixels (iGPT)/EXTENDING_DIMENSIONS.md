## 1. Basic Metadata

- Title: Generative Pretraining from Pixels.
  - Evidence: "Generative Pretraining from Pixels" (Title block)
- Authors: Mark Chen; Alec Radford; Rewon Child; Jeff Wu; Heewoo Jun; Prafulla Dhariwal; David Luan; Ilya Sutskever.
  - Evidence: "Mark Chen <sup>1</sup> Alec Radford <sup>1</sup> Rewon Child <sup>1</sup> Jeff Wu <sup>1</sup> Heewoo Jun <sup>1</sup> Prafulla Dhariwal <sup>1</sup> David Luan <sup>1</sup> Ilya Sutskever <sup>1</sup>" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper examines generative pretraining for images by training a sequence Transformer to predict pixels and then evaluating the learned representations via linear probing and fine-tuning for image classification.

Evidence:
- "We train a sequence Transformer to auto-regressively predict pixels, without incorporating knowledge of the 2D input structure." (Abstract)
- "One way to measure representation quality is to fine-tune for image classification." (Section 2)
- "Another approach for measuring representation quality uses the pre-trained model as a feature extractor. In particular, given labeled examples (X, Y), the model is applied to X to produce features  $f_X$ . Then, a linear classifier is trained on  $(f_X, Y)$ ." (Section 2)

## 3. Tasks Evaluated

- Task name: Autoregressive next-pixel prediction (pre-training)
  - Task type: Generation
  - Dataset(s) used: ImageNet ILSVRC 2012 training dataset; additional 100 million unlabeled web images (largest model)
  - Domain: Images (ImageNet, web images)
  - Quotes:
    - "We train a sequence Transformer to auto-regressively predict pixels, without incorporating knowledge of the 2D input structure." (Abstract)
    - "We use the ImageNet ILSVRC 2012 training dataset" (Section 3.1)
    - "For our largest model, we use an additional 100 million unlabeled web images, filtered to be similar to ImageNet." (Section 3)

- Task name: Masked pixel prediction (BERT objective, pre-training)
  - Task type: Reconstruction (masked prediction)
  - Dataset(s) used: ImageNet ILSVRC 2012 training dataset
  - Domain: Images
  - Quotes:
    - "We also consider the BERT objective, which samples a sub-sequence  $M \subset [1,n]$  such that each index i independently has probability 0.15 of appearing in M." (Section 2.1)
    - "We use the ImageNet ILSVRC 2012 training dataset" (Section 3.1)

- Task name: Image classification via fine-tuning
  - Task type: Classification
  - Dataset(s) used: CIFAR-10, CIFAR-100, ImageNet
  - Domain: Images
  - Quotes:
    - "One way to measure representation quality is to fine-tune for image classification." (Section 2)
    - "On CIFAR-10, iGPT-L achieves 99.0% accuracy and on CIFAR-100, it achieves 88.5% accuracy after fine-tuning." (Section 4.5)
    - "On ImageNet, we achieve 66.3% accuracy after fine-tuning" (Section 4.5)

- Task name: Image classification via linear probing (feature extractor)
  - Task type: Classification
  - Dataset(s) used: CIFAR-10, CIFAR-100, STL-10, ImageNet
  - Domain: Images
  - Quotes:
    - "Another approach for measuring representation quality uses the pre-trained model as a feature extractor." (Section 2)
    - "In addition to CIFAR-10, we also evaluate linear probes on CIFAR-100 and STL-10" (Section 4.3)
    - "Recently, there has been a resurgence of interest in unsupervised and self-supervised learning on ImageNet, evaluated using linear probes on ImageNet." (Section 4.4)

- Task name: Low-data CIFAR-10 classification
  - Task type: Classification
  - Dataset(s) used: CIFAR-10 (subset / low-data regime)
  - Domain: Images
  - Quotes:
    - "### 4.7. Low-Data CIFAR-10 Classification" (Section 4.7)
    - "This motivates evaluating performance in a low-data regime as well." (Section 4.7)
    - "we work directly on a subset of the raw supervised dataset, extracting features using our pre-trained model, and training a linear classifier on those features." (Section 4.7)

## 4. Domain and Modality Scope

- Evaluation is performed on multiple datasets within the same modality (images).
  - Evidence: "We investigate this setting using ImageNet as a proxy for a large unlabeled corpus, and small classic labeled datasets (CIFAR-10, CIFAR-100, STL-10) as proxies for downstream tasks." (Section 3)
- Modality: Single modality (images).
  - Evidence: "we examine whether similar models can learn useful representations for images." (Abstract)
- Domain generalization or cross-domain transfer: Not claimed.
  - Evidence: The paper frames transfer within image datasets: "ImageNet as a proxy for a large unlabeled corpus, and small classic labeled datasets (CIFAR-10, CIFAR-100, STL-10) as proxies for downstream tasks." (Section 3)

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Autoregressive next-pixel prediction | Trained separately per objective | No | Yes (projection to logits) | "In pre-training, we pick one of  $L_{AR}$  or  $L_{BERT}$  and minimize the loss" (Section 2.1); "learn a projection from  $n^L$  to logits parameterizing the conditional distributions" (Section 2.2) |
| Masked pixel prediction (BERT) | Trained separately per objective | No | Yes (projection to logits) | "We also consider the BERT objective" (Section 2.1); "When training BERT, we simply ignore the logits at unmasked positions." (Section 2.2) |
| Image classification via linear probing | Yes (same pre-trained model as feature extractor) | No (features fixed) | Yes (linear classifier) | "the model is applied to X to produce features  $f_X$ . Then, a linear classifier is trained" (Section 2); "Because we view the features as fixed when linear probing, this projection contains the only trainable weights" (Section 2.4) |
| Image classification via fine-tuning | Pretrained once, then fine-tuned per task | Yes | Yes (classification head) | "Our approach consists of a pre-training stage followed by a fine-tuning stage." (Section 2); "Fine-tuning adds a small classification head to the model, used to optimize a classification objective and adapts all weights." (Section 2.3) |
| Low-data CIFAR-10 classification | Yes (pre-trained model used as feature extractor) | No | Yes (linear classifier) | "extracting features using our pre-trained model, and training a linear classifier on those features." (Section 4.7) |

## 6. Input and Representation Constraints

- Images are resized to low resolution and reshaped into a 1D sequence.
  - Evidence: "First, we pre-process raw images by resizing to a low resolution and reshaping into a 1D sequence." (Figure 1 caption)
- Fixed input resolutions per model:  $32^2 \times 3$ ,  $48^2 \times 3$ , or  $64^2 \times 3$ .
  - Evidence: "Our models have IRs of either  $32^2 \times 3$ ,  $48^2 \times 3$ , or  $64^2 \times 3$ ." (Section 3.2)
- Fixed context length (model resolution):  $32^2$ ,  $48^2$ , or  $64^2$ tokens.
  - Evidence: "We call the resulting context length ( $32^2$  or  $48^2$  or  $64^2$ ) the model resolution (MR)." (Section 3.2)
- Raster order tokenization (fixed ordering).
  - Evidence: "When working with images, we pick the identity permutation  $\pi_i = i$  for  $1 \le i \le n$ , also known as raster order." (Section 2.1)
- Quantized color palette (512 colors) to reduce sequence length.
  - Evidence: "we create our own 9-bit color palette by clustering (R, G, B) pixel values using k-means with k=512. Using this palette yields an input sequence length 3 times shorter" (Section 3.2)
- Patch size: Not specified; inputs are pixel-level tokens.
  - Evidence: "We train a sequence Transformer to auto-regressively predict pixels" (Abstract); "The transformer decoder takes an input sequence  $x_1, ..., x_n$  of discrete tokens" (Section 2.2)
- The model does not encode 2D structure in its architecture.
  - Evidence: "without incorporating knowledge of the 2D input structure." (Abstract)
- Resizing/cropping requirements for ImageNet evaluation and training.
  - Evidence: "we randomly resize an image such that the shorter sidelength is in the range [256, 384] and then take a random  $224 \times 224$  crop. When evaluating on ImageNet, we resize the image such that the shorter sidelength is 224, and use the single  $224 \times 224$  center crop." (Section 3.1)

## 7. Context Window and Attention Structure

- Maximum sequence length / context length: up to  $64^2$ tokens (MR).
  - Evidence: "We call the resulting context length ( $32^2$  or  $48^2$  or  $64^2$ ) the model resolution (MR)." (Section 3.2)
- Sequence length is fixed per model resolution (not variable within a run).
  - Evidence: "Our models have IRs of either  $32^2 \times 3$ ,  $48^2 \times 3$ , or  $64^2 \times 3$ ." (Section 3.2)
- Attention type: global dense self-attention with causal mask for autoregressive training; unmasked for BERT.
  - Evidence: "memory requirements of the transformer decoder scale quadratically with context length when using dense attention" (Section 3.2); "we apply the standard upper triangular mask to the  $n \times n$  matrix of attention logits" (Section 2.2); "When using the BERT objective, no attention logit masking is required" (Section 2.2)
- Computational cost mitigation: context reduction via lower resolution and palette quantization.
  - Evidence: "we first resize our image to a lower resolution" (Section 3.2); "Using this palette yields an input sequence length 3 times shorter" (Section 3.2)

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: learned independent position embeddings per sequence element (absolute).
  - Evidence: "since we learn independent position embeddings for each sequence element" (Section 2.2)
- Where it is applied: Not explicitly stated beyond per-position embeddings in the input sequence.
  - Evidence: "since we learn independent position embeddings for each sequence element" (Section 2.2)
- Fixed vs modified/ablated: Not specified; no comparisons described.

## 9. Positional Encoding as a Variable

- Core research variable vs fixed assumption: Treated as a fixed architectural assumption; no explicit experimentation on PE.
  - Evidence: "since we learn independent position embeddings for each sequence element" (Section 2.2)
- Multiple positional encodings compared: Not stated.
- Claim that PE is not critical or secondary: Not stated.

## 10. Evidence of Constraint Masking

- Model sizes are large and explicitly scaled:
  - "Our largest model, iGPT-XL, contains L=60 layers and uses an embedding size of d=3072 for a total of 6.8B parameters." (Section 3.3)
  - "We also train iGPT-M, a 455M parameter model with L=36 and d=1024 and iGPT-S, a 76M parameter model with L=24 and d=512 to study the effect of model capacity on representation quality in a generative model." (Section 3.3)
- Dataset sizes include ImageNet and an additional 100M web images:
  - "We use the ImageNet ILSVRC 2012 training dataset" (Section 3.1)
  - "For our largest model, we use an additional 100 million unlabeled web images" (Section 3)
- Performance gains attributed to scaling model size:
  - "This highlights the importance of scale for our approach." (Section 4.2)
  - "our approach requires large models in order to learn high quality representations. iGPT-L has 2 to 3 times as many parameters as similarly performing models on ImageNet and uses more compute." (Section 6)
- Performance gains attributed to scaling data or architectural hierarchy: Not explicitly stated.

## 11. Architectural Workarounds

- Context reduction via lower resolution inputs to manage dense attention cost.
  - Evidence: "we first resize our image to a lower resolution, which we call the input resolution (IR)." (Section 3.2)
- Token reduction via 9-bit color palette (k=512) to shorten sequence length.
  - Evidence: "we create our own 9-bit color palette by clustering (R, G, B) pixel values using k-means with k=512. Using this palette yields an input sequence length 3 times shorter" (Section 3.2)
- Causal masking for autoregressive training; unmasked for BERT.
  - Evidence: "we apply the standard upper triangular mask to the  $n \times n$  matrix of attention logits" (Section 2.2); "When using the BERT objective, no attention logit masking is required" (Section 2.2)
- Task-specific classification head and pooling.
  - Evidence: "Fine-tuning adds a small classification head to the model" (Section 2.3); "we average pool  $n^L$  across the sequence dimension to extract a d-dimensional vector of features" (Section 2.3)
- Fixed 1D raster order of pixels.
  - Evidence: "also known as raster order." (Section 2.1)

## 12. Explicit Limitations and Non-Claims

- Low-resolution inputs and gap to high-resolution approaches:
  - "We currently model low resolution inputs with self-attention. By comparison, most other self-supervised results use CNN based encoders that easily work with high resolution images." (Section 6)
  - "It is not immediately obvious how to best bridge the gap between performant autoregressive and discriminative models." (Section 6)
- Large model and compute requirements:
  - "our approach requires large models in order to learn high quality representations. iGPT-L has 2 to 3 times as many parameters as similarly performing models on ImageNet and uses more compute." (Section 6)
- Dense self-attention is a significant limitation despite context reduction:
  - "Although dense self-attention was a deliberate choice for this work due to it being domain agnostic and widely used in NLP, it becomes very memory and computationally expensive due to its quadratic scaling with sequence length. We mitigated this via the context reduction techniques discussed in section 3.2 but it is still a significant limitation." (Section 6)
- Explicit non-claims in low-data setting:
  - "we do not make use of pseudo-labeling or data augmentation." (Section 4.7)
- Explicit statements about open-world learning, unrestrained multi-task learning, or meta-learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Single modality (images) with multiple image datasets (ImageNet, CIFAR-10/100, STL-10).
> - Task structure: Pretraining on pixel prediction; downstream evaluation is classification via linear probes and fine-tuning.
> - Representation rigidity: Fixed low-resolution inputs, raster order 1D sequences, fixed context length (MR).
> - Model sharing vs specialization: Pretrain once, then separate linear heads or fine-tuning per dataset/task; no joint multi-task training.
> - Role of positional encoding: Learned per-position embeddings, treated as fixed and not compared.

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple downstream image classification settings across CIFAR-10, CIFAR-100, STL-10, and ImageNet while staying within the image modality. This is framed as transfer from ImageNet pretraining to multiple image datasets rather than cross-modal or open-world tasks ("We investigate this setting using ImageNet as a proxy for a large unlabeled corpus, and small classic labeled datasets (CIFAR-10, CIFAR-100, STL-10) as proxies for downstream tasks." (Section 3)).
