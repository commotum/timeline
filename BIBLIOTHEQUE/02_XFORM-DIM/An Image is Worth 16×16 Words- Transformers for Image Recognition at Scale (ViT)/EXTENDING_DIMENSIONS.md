## 1. Basic Metadata

- Title: "AN IMAGE IS WORTH 16x16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE" (Title block)
- Authors: "Alexey Dosovitskiy\*,†, Lucas Beyer\*, Alexander Kolesnikov\*, Dirk Weissenborn\*, Xiaohua Zhai\*, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby\*,†" (Title block)
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

The paper claims that "a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks." (Abstract)

## 3. Tasks Evaluated

| Task name | Task type | Dataset(s) used | Domain | Evidence (quote) |
| --- | --- | --- | --- | --- |
| Image classification (ImageNet validation labels) | Classification | ILSVRC-2012 ImageNet | images | "We use the ILSVRC-2012 ImageNet dataset with 1k classes and 1.3M images (we refer to it as ImageNet in what follows), its superset ImageNet-21k with 21k classes and 14M images (Deng et al., 2009), and JFT (Sun et al., 2017) with 18k classes and 303M high-resolution images." (Section 4.1 Setup) / "We transfer the models trained on these dataset to several benchmark tasks: ImageNet on the original validation labels and the cleaned-up ReaL labels (Beyer et al., 2020), CIFAR-10/100 (Krizhevsky, 2009), Oxford-IIIT Pets (Parkhi et al., 2012), and Oxford Flowers-102 (Nilsback & Zisserman, 2008)." (Section 4.1 Setup) |
| Image classification (ImageNet ReaL labels) | Classification | ImageNet ReaL | images | "We use the ILSVRC-2012 ImageNet dataset with 1k classes and 1.3M images (we refer to it as ImageNet in what follows), its superset ImageNet-21k with 21k classes and 14M images (Deng et al., 2009), and JFT (Sun et al., 2017) with 18k classes and 303M high-resolution images." (Section 4.1 Setup) / "ImageNet on the original validation labels and the cleaned-up ReaL labels (Beyer et al., 2020)" (Section 4.1 Setup) |
| Image classification (CIFAR-10) | Classification | CIFAR-10 | Not specified. | "CIFAR-10/100 (Krizhevsky, 2009)" (Section 4.1 Setup) |
| Image classification (CIFAR-100) | Classification | CIFAR-100 | Not specified. | "CIFAR-10/100 (Krizhevsky, 2009)" (Section 4.1 Setup) |
| Image classification (Oxford-IIIT Pets) | Classification | Oxford-IIIT Pets | Not specified. | "Oxford-IIIT Pets (Parkhi et al., 2012)" (Section 4.1 Setup) |
| Image classification (Oxford Flowers-102) | Classification | Oxford Flowers-102 | Not specified. | "Oxford Flowers-102 (Nilsback & Zisserman, 2008)" (Section 4.1 Setup) |
| VTAB classification suite (19 tasks) | Classification | VTAB (19-task classification suite) | Multiple image domains (Natural, Specialized, Structured) | "We also evaluate on the 19-task VTAB classification suite (Zhai et al., 2019b). VTAB evaluates low-data transfer to diverse tasks, using 1 000 training examples per task. The tasks are divided into three groups: *Natural* – tasks like the above, Pets, CIFAR, etc. *Specialized* – medical and satellite imagery, and *Structured* – tasks that require geometric understanding like localization." (Section 4.1 Setup) |
| Image classification (ObjectNet benchmark) | Classification | ObjectNet benchmark | Not specified. | "We also evaluate our flagship ViT-H/14 model on the ObjectNet benchmark following the evaluation setup in Kolesnikov et al. (2020), resulting in 82.1% top-5 accuracy and 61.7% top-1 accuracy." (Appendix D.9 ObjectNet Results) |
| Masked patch prediction (self-supervised pretraining) | Reconstruction | JFT | high-resolution images | "We employ the *masked patch prediction* objective for preliminary self-supervision experiments." (Appendix B.1.2 Self-supervision) / "We trained our self-supervised model for 1M steps (ca. 14 epochs) with batch size 4096 on JFT." (Appendix B.1.2 Self-supervision) / "JFT (Sun et al., 2017) with 18k classes and 303M high-resolution images." (Section 4.1 Setup) |

## 4. Domain and Modality Scope

- Evaluation scope: Multiple domains within the same modality (images). Evidence: "We split an image into fixed-size patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder." (Figure 1 caption) and "The tasks are divided into three groups: *Natural* – tasks like the above, Pets, CIFAR, etc. *Specialized* – medical and satellite imagery, and *Structured* – tasks that require geometric understanding like localization." (Section 4.1 Setup)
- Multiple modalities? Not claimed.
- Domain generalization or cross-domain transfer? Yes; "When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.<sup>1</sup>" (Abstract) / "We transfer the models trained on these dataset to several benchmark tasks: ImageNet on the original validation labels and the cleaned-up ReaL labels (Beyer et al., 2020), CIFAR-10/100 (Krizhevsky, 2009), Oxford-IIIT Pets (Parkhi et al., 2012), and Oxford Flowers-102 (Nilsback & Zisserman, 2008)." (Section 4.1 Setup)

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| ImageNet (validation labels) | Yes (pretrained model transferred) | Yes | Yes | "Typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks. For this, we remove the pre-trained prediction head and attach a zero-initialized  $D \times K$  feedforward layer, where K is the number of downstream classes." (Section 3.2 Fine-Tuning and Higher Resolution) / "We transfer the models trained on these dataset to several benchmark tasks: ImageNet on the original validation labels and the cleaned-up ReaL labels (Beyer et al., 2020), CIFAR-10/100 (Krizhevsky, 2009), Oxford-IIIT Pets (Parkhi et al., 2012), and Oxford Flowers-102 (Nilsback & Zisserman, 2008)." (Section 4.1 Setup) |
| ImageNet ReaL | Yes (pretrained model transferred) | Yes | Yes | "Typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks. For this, we remove the pre-trained prediction head and attach a zero-initialized  $D \times K$  feedforward layer, where K is the number of downstream classes." (Section 3.2 Fine-Tuning and Higher Resolution) / "When transferring ViT models to another dataset, we remove the whole head (two linear layers) and replace it by a single, zero-initialized linear layer outputting the number of classes required by the target dataset." (Appendix B.1.1 Fine-Tuning) |
| CIFAR-10 | Yes (pretrained model transferred) | Yes | Yes | "Typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks." (Section 3.2 Fine-Tuning and Higher Resolution) / "When transferring ViT models to another dataset, we remove the whole head (two linear layers) and replace it by a single, zero-initialized linear layer outputting the number of classes required by the target dataset." (Appendix B.1.1 Fine-Tuning) |
| CIFAR-100 | Yes (pretrained model transferred) | Yes | Yes | "Typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks." (Section 3.2 Fine-Tuning and Higher Resolution) / "When transferring ViT models to another dataset, we remove the whole head (two linear layers) and replace it by a single, zero-initialized linear layer outputting the number of classes required by the target dataset." (Appendix B.1.1 Fine-Tuning) |
| Oxford-IIIT Pets | Yes (pretrained model transferred) | Yes | Yes | "Typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks." (Section 3.2 Fine-Tuning and Higher Resolution) / "When transferring ViT models to another dataset, we remove the whole head (two linear layers) and replace it by a single, zero-initialized linear layer outputting the number of classes required by the target dataset." (Appendix B.1.1 Fine-Tuning) |
| Oxford Flowers-102 | Yes (pretrained model transferred) | Yes | Yes | "Typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks." (Section 3.2 Fine-Tuning and Higher Resolution) / "When transferring ViT models to another dataset, we remove the whole head (two linear layers) and replace it by a single, zero-initialized linear layer outputting the number of classes required by the target dataset." (Appendix B.1.1 Fine-Tuning) |
| VTAB (19 tasks) | Yes (pretrained model transferred) | Yes | Yes | "Typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks." (Section 3.2 Fine-Tuning and Higher Resolution) / "VTAB evaluates low-data transfer to diverse tasks, using 1 000 training examples per task." (Section 4.1 Setup) / "When transferring ViT models to another dataset, we remove the whole head (two linear layers) and replace it by a single, zero-initialized linear layer outputting the number of classes required by the target dataset." (Appendix B.1.1 Fine-Tuning) |
| ObjectNet benchmark | Yes (pretrained model transferred) | Yes | Yes | "Typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks." (Section 3.2 Fine-Tuning and Higher Resolution) / "When transferring ViT models to another dataset, we remove the whole head (two linear layers) and replace it by a single, zero-initialized linear layer outputting the number of classes required by the target dataset." (Appendix B.1.1 Fine-Tuning) |
| Masked patch prediction (self-supervised pretraining) | Not specified. | Not specified. | Not specified. | "We employ the *masked patch prediction* objective for preliminary self-supervision experiments." (Appendix B.1.2 Self-supervision) |

## 6. Input and Representation Constraints

- Fixed-size patches and patchified input: "We split an image into fixed-size patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder." (Figure 1 caption)
- Patch size and sequence length formula: "(P, P) is the resolution of each image patch, and  $N = HW/P^2$  is the resulting number of patches, which also serves as the effective input sequence length for the Transformer." (Section 3.1 Vision Transformer)
- Fixed embedding dimensionality: "The Transformer uses constant latent vector size D through all of its layers" (Section 3.1 Vision Transformer)
- Variable resolution / variable sequence length: "When feeding images of higher resolution, we keep the patch size the same, which results in a larger effective sequence length. The Vision Transformer can handle arbitrary sequence lengths (up to memory constraints)" (Section 3.2 Fine-Tuning and Higher Resolution)
- Training resolution: "Finally, all training is done on resolution 224." (Appendix B.1 Training)
- Fine-tuning resolution and resizing policy: "If not mentioned otherwise, fine-tuning resolution is 384." (Table 4) / "we do not use task-specific input resolutions. Instead we find that Vision Transformer benefits most from a high resolution ( $384 \times 384$ ) for all tasks." (Appendix B.1.1 Fine-Tuning)
- Position embedding interpolation for resolution change: "We therefore perform 2D interpolation of the pre-trained position embeddings, according to their location in the original image." (Section 3.2 Fine-Tuning and Higher Resolution)
- Padding requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; "The Vision Transformer can handle arbitrary sequence lengths (up to memory constraints)" (Section 3.2 Fine-Tuning and Higher Resolution)
- Sequence length fixed or variable: Variable with input resolution; "When feeding images of higher resolution, we keep the patch size the same, which results in a larger effective sequence length." (Section 3.2 Fine-Tuning and Higher Resolution)
- Attention type: Global; "the self-attention layers are global." (Section 3.1 Inductive bias)
- Cost management mechanisms: Patch size controls sequence length and compute; "Note that the Transformer's sequence length is inversely proportional to the square of the patch size, thus models with smaller patch size are computationally more expensive." (Section 4.1 Model Variants)

## 8. Positional Encoding (Critical Section)

- Mechanism: "Position embeddings are added to the patch embeddings to retain positional information. We use standard learnable 1D position embeddings, since we have not observed significant performance gains from using more advanced 2D-aware position embeddings (Appendix D.4)." (Section 3.1 Vision Transformer)
- Where applied: "add positional embeddings to the inputs right after the stem of them model and before feeding the inputs to the Transformer encoder (default across all other experiments in this paper)" (Appendix D.4 Positional Embedding)
- Alternatives compared: "Providing no positional information: Considering the inputs as a *bag of patches*." / "1-dimensional positional embedding: Considering the inputs as a sequence of patches in the raster order (default across all other experiments in this paper)." / "2-dimensional positional embedding: Considering the inputs as a grid of patches in two dimensions." / "Relative positional embeddings: Considering the relative distance between patches to encode the spatial information as instead of their absolute position." (Appendix D.4 Positional Embedding)
- Resolution changes: "We therefore perform 2D interpolation of the pre-trained position embeddings, according to their location in the original image." (Section 3.2 Fine-Tuning and Higher Resolution)

## 9. Positional Encoding as a Variable

- Core research variable vs fixed assumption: Fixed in main experiments but ablated; "1-dimensional positional embedding: Considering the inputs as a sequence of patches in the raster order (default across all other experiments in this paper)." (Appendix D.4 Positional Embedding) and "We ran ablations on different ways of encoding spatial information using positional embedding." (Appendix D.4 Positional Embedding)
- Multiple positional encodings compared: "Providing no positional information: Considering the inputs as a *bag of patches*." / "1-dimensional positional embedding: Considering the inputs as a sequence of patches in the raster order (default across all other experiments in this paper)." / "2-dimensional positional embedding: Considering the inputs as a grid of patches in two dimensions." / "Relative positional embeddings: Considering the relative distance between patches to encode the spatial information as instead of their absolute position." (Appendix D.4 Positional Embedding)
- PE criticality: "there is little to no difference between different ways of encoding positional information." (Appendix D.4 Positional Embedding)

## 10. Evidence of Constraint Masking

- Model sizes: "| ViT-Base  | 12     | 768                     | 3072     | 12    | 86M    |" / "| ViT-Large | 24     | 1024                    | 4096     | 16    | 307M   |" / "| ViT-Huge  | 32     | 1280                    | 5120     | 16    | 632M   |" (Table 1)
- Dataset sizes: "We use the ILSVRC-2012 ImageNet dataset with 1k classes and 1.3M images (we refer to it as ImageNet in what follows), its superset ImageNet-21k with 21k classes and 14M images (Deng et al., 2009), and JFT (Sun et al., 2017) with 18k classes and 303M high-resolution images." (Section 4.1 Setup)
- Scaling data vs inductive bias: "However, the picture changes if the models are trained on larger datasets (14M-300M images). We find that large scale training trumps inductive bias." (Introduction)
- Scaling model size with large data: "Only with JFT-300M, do we see the full benefit of larger models." (Section 4.3 Pre-training Data Requirements)

## 11. Architectural Workarounds

- Patchify images into tokens: "We split an image into fixed-size patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder." (Figure 1 caption)
- Classification token and head: "In order to stay as close as possible to the original Transformer model, we made use of an additional <code>[class]</code> token, which is taken as image representation. The output of this token is then transformed into a class prediction via a small multi-layer perceptron (MLP) with tanh as non-linearity in the single hidden layer." (Appendix D.3 Head Type and Class Token)
- Hybrid CNN+ViT option: "As an alternative to raw image patches, the input sequence can be formed from feature maps of a CNN (LeCun et al., 1989)." (Section 3.1 Vision Transformer)
- Task-specific head replacement for transfer: "we remove the pre-trained prediction head and attach a zero-initialized  $D \times K$  feedforward layer, where K is the number of downstream classes." (Section 3.2 Fine-Tuning and Higher Resolution)
- Resolution adjustment via PE interpolation: "We therefore perform 2D interpolation of the pre-trained position embeddings, according to their location in the original image." (Section 3.2 Fine-Tuning and Higher Resolution)

## 12. Explicit Limitations and Non-Claims

- "Vision Transformers overfit more than ResNets with comparable computational cost on smaller datasets." (Section 4.3 Pre-training Data Requirements)
- "We leave exploration of contrastive pre-training (Chen et al., 2020b; He et al., 2020; Bachman et al., 2019; Hénaff et al., 2020) to future work." (Section 4.6 Self-supervision)
- "One is to apply ViT to other computer vision tasks, such as detection and segmentation." (Conclusion)
- "Our initial experiments show improvement from self-supervised pre-training, but there is still large gap between self-supervised and large-scale supervised pre-training." (Conclusion)
- No explicit statements about open-world learning or unrestrained multi-task learning were found.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: multiple image domains within a single modality, via transfer to varied benchmarks and VTAB groups
> – Task structure: supervised image classification across datasets plus a constrained self-supervised masked patch prediction pretraining task
> – Representation rigidity: fixed-size patches and fixed D, fixed training/fine-tuning resolutions with interpolated position embeddings when resolution changes
> – Model sharing vs specialization: shared pretrained backbone, fine-tuned per dataset with a new classification head
> – Role of positional encoding: learnable 1D absolute PE by default, with ablations showing limited sensitivity

### 14. Final Classification

**Multi-task, multi-domain (constrained).** The paper transfers to multiple benchmarks and evaluates a "19-task VTAB classification suite" with domains including "medical and satellite imagery" and "tasks that require geometric understanding like localization," indicating multiple domains within a single modality. At the same time, it uses a pretrain-then-fine-tune setup with task-specific heads ("pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks"), so the multi-task setting is constrained rather than unrestrained.
