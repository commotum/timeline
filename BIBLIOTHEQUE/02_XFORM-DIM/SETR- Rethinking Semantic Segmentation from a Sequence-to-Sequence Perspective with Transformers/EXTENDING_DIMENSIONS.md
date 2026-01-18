## 1. Basic Metadata

- Title: "Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers" (Title block)
- Authors: "Sixiao Zheng<sup>1\*</sup> Jiachen Lu<sup>1</sup> Hengshuang Zhao<sup>2</sup> Xiatian Zhu<sup>3</sup> Zekun Luo<sup>4</sup> Yabiao Wang<sup>4</sup> Yanwei Fu<sup>1</sup> Jianfeng Feng<sup>1</sup> Tao Xiang<sup>3, 5</sup> Philip H.S. Torr<sup>2</sup> Li Zhang<sup>1†</sup>" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

"In this paper, we aim to provide an alternative perspective by treating semantic segmentation as a sequence-to-sequence prediction task." (Abstract)

---

## 3. Tasks Evaluated

### Task 1: Semantic segmentation (Cityscapes)
- Task type: Segmentation
- Dataset(s) used: Cityscapes
- Domain: Natural images (urban scenes)
- Quotes: "We conduct experiments on three widely-used semantic segmentation benchmark datasets." (4.1. Experimental setup) "Cityscapes [13] densely annotates 19 object categories in images with urban scenes." (4.1. Experimental setup)

### Task 2: Semantic segmentation (ADE20K)
- Task type: Segmentation
- Dataset(s) used: ADE20K
- Domain: Natural images (scene parsing)
- Quotes: "We conduct experiments on three widely-used semantic segmentation benchmark datasets." (4.1. Experimental setup) "**ADE20K** [63] is a challenging scene parsing benchmark with 150 fine-grained semantic concepts." (4.1. Experimental setup)

### Task 3: Semantic segmentation (PASCAL Context)
- Task type: Segmentation
- Dataset(s) used: PASCAL Context
- Domain: Natural images (scene parsing)
- Quotes: "We conduct experiments on three widely-used semantic segmentation benchmark datasets." (4.1. Experimental setup) "**PASCAL Context** [37] provides pixel-wise semantic labels for the whole scene (both \"thing\" and \"stuff\" classes)." (4.1. Experimental setup)

---

## 4. Domain and Modality Scope

- Evaluation scope: "We conduct experiments on three widely-used semantic segmentation benchmark datasets." (4.1. Experimental setup)
- Domains/modality: Single modality (images) across multiple datasets within the same modality, including "images with urban scenes" and scene parsing datasets. (4.1. Experimental setup: "Cityscapes [13] densely annotates 19 object categories in images with urban scenes."; "**ADE20K** [63] is a challenging scene parsing benchmark..."; "**PASCAL Context** [37] provides pixel-wise semantic labels for the whole scene...")
- Domain generalization or cross-domain transfer: Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Semantic segmentation (Cityscapes) | Not specified. | Pretrained initialization stated; fine-tuning per task not explicitly stated. | Not specified. | "We use the pre-trained weights provided by ViT [17] or DeiT [44] to initialize all the transformer layers and the input linear projection layer in our model." (4.1. Experimental setup) "For Cityscapes, we set batch size to 8 with a number of training schedules reported in Table 2, 6 and 7 for fair comparison." (4.1. Experimental setup) |
| Semantic segmentation (ADE20K) | Not specified. | Pretrained initialization stated; fine-tuning per task not explicitly stated. | Not specified. | "We use the pre-trained weights provided by ViT [17] or DeiT [44] to initialize all the transformer layers and the input linear projection layer in our model." (4.1. Experimental setup) "We set batch size 16 and the total iteration to 160,000 and 80,000 for the experiments on ADE20K and Pascal Context." (4.1. Experimental setup) |
| Semantic segmentation (PASCAL Context) | Not specified. | Pretrained initialization stated; fine-tuning per task not explicitly stated. | Not specified. | "We use the pre-trained weights provided by ViT [17] or DeiT [44] to initialize all the transformer layers and the input linear projection layer in our model." (4.1. Experimental setup) "We set batch size 16 and the total iteration to 160,000 and 80,000 for the experiments on ADE20K and Pascal Context." (4.1. Experimental setup) |

---

## 6. Input and Representation Constraints

- Input is a 2D RGB image tensor: "the first layer takes as input the image, denoted as  $H \times W \times 3$  with  $H \times W$  specifying the image size in pixels." (3.1. FCN-based semantic segmentation)
- Fixed patch grid and patch size: "we first decompose an image into a grid of fixed-sized patches" (Introduction) and "To obtain the  $\frac{HW}{256}$ -long input sequence, we divide an image  $x \in \mathbb{R}^{H \times W \times 3}$  into a grid of  $\frac{H}{16} \times \frac{W}{16}$  patches uniformly" (3.2. Segmentation transformers) and "We use patch size  $16 \times 16$  for all the experiments." (4.1. Experimental setup)
- Fixed token count per input size: "we thus decide to set the transformer input sequence length L as  $\frac{H}{16} \times \frac{W}{16} = \frac{HW}{256}$ ." (3.2. Segmentation transformers)
- Positional embedding added to each patch embedding: "we learn a specific embedding  $p_i$  for every location i which is added to  $e_i$  to form the final sequence input" (3.2. Segmentation transformers)
- Training-time resizing/cropping constraints: "we apply random resize with ratio between 0.5 and 2, random cropping (768, 512 and 480 for Cityscapes, ADE20K and Pascal Context respectively) and random horizontal flipping during training for all the experiments" (4.1. Experimental setup)
- Test-time resizing and sliding window: "Sliding window is adopted for test  $(e.g., 480 \times 480 \text{ for Pascal Context})$ ." (4.1. Experimental setup)
- Output stride constraint: "we use output stride 16 in all our models due to GPU memory constrain." (4.1. Experimental setup)

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not explicitly specified; sequence length is defined by input size as "L" where "we thus decide to set the transformer input sequence length L as  $\frac{H}{16} \times \frac{W}{16} = \frac{HW}{256}$ ." (3.2. Segmentation transformers)
- Fixed or variable length: Variable with input size based on " $\frac{H}{16} \times \frac{W}{16}$ " patches. (3.2. Segmentation transformers)
- Attention type: Global self-attention: "With the global context modeled in every layer of the transformer" (Abstract) and "This means each transformer layer has a global receptive field" (3.2. Segmentation transformers)
- Mechanisms to manage computational cost: "Given the quadratic model complexity of Transformer, it is not possible that such high-dimensional vectors can be handled in both space and time. Therefore tokenizing every single pixel as input to our transformer is out of the question." (3.2. Segmentation transformers) and "we thus decide to set the transformer input sequence length L as  $\frac{H}{16} \times \frac{W}{16} = \frac{HW}{256}$ ." (3.2. Segmentation transformers)

---

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: Learned absolute position embeddings per patch: "we learn a specific embedding  $p_i$  for every location i which is added to  $e_i$  to form the final sequence input  $E = \{e_1 + p_1, \ e_2 + p_2, \ \cdots, \ e_L + p_L\}$ ." (3.2. Segmentation transformers)
- Where it is applied: Input only (added to patch embeddings before the transformer): "which is added to  $e_i$  to form the final sequence input" (3.2. Segmentation transformers)
- Fixed vs modified: Position embeddings are used consistently with interpolation for different input sizes: "We perform 2D interpolation on the pre-trained position embeddings, according to their location in the original image for different input size fine-tuning." (4.1. Experimental setup)

---

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption: Positional encoding is presented as a fixed architectural component; no ablation or comparison is stated. Evidence of fixed usage: "we learn a specific embedding  $p_i$  for every location i which is added to  $e_i$  to form the final sequence input" (3.2. Segmentation transformers) and "We perform 2D interpolation on the pre-trained position embeddings..." (4.1. Experimental setup)
- Multiple positional encodings compared: Not stated.
- Claim that PE choice is "not critical" or secondary: Not stated.

---

## 10. Evidence of Constraint Masking

- Model sizes: "T-Base  | 12       | 768         | 12" and "T-Large | 24       | 1024        | 16" (Table 1. Configuration of Transformer backbone variants) and "SETR-PUP          | 21K | T-Large  | 318.31M" (Table 2)
- Dataset sizes: "Cityscapes [13]... It contains 5000 finely annotated images, split into 2975, 500 and 1525 for training, validation and testing respectively." (4.1. Experimental setup) "**ADE20K** [63]... It contains 20210, 2000 and 3352 images for training, validation and testing." (4.1. Experimental setup) "**PASCAL Context** [37]... contains 4998 and 5105 images for training and validation respectively." (4.1. Experimental setup)
- Scaling model size: "The variants using \"T-Large\" (e.g., SETR-MLA and SETR-Naïve) are superior to their \"T-Base\" counterparts" (4.2. Ablation studies)
- Scaling data / pretraining: "ImageNet-21k pre-training FCN baseline experienced a clear improvement over the variant pre-trained on ImageNet-1k." (4.2. Ablation studies) and "our method outperforms the FCN counterparts by a large margin, verifying that the advantage of our approach largely comes from the proposed sequence-to-sequence modeling strategy rather than bigger pre-training data." (4.2. Ablation studies)
- Training tricks and initialization: "Pre-training is critical for our model. Randomly initialized SETR-PUP only gives 42.27% mIoU on Cityscapes." (4.2. Ablation studies) and "For training simplicity, we do not adopt the widely-used tricks such as OHEM [55] loss in model training." (4.1. Experimental setup)

---

## 11. Architectural Workarounds

- Patch tokenization to reduce sequence length: "we thus decide to set the transformer input sequence length L as  $\frac{H}{16} \times \frac{W}{16} = \frac{HW}{256}$ ." (3.2. Segmentation transformers)
- Fixed grid assumption for 2D images: "we divide an image  $x \in \mathbb{R}^{H \times W \times 3}$  into a grid of  $\frac{H}{16} \times \frac{W}{16}$  patches uniformly" (3.2. Segmentation transformers)
- Progressive upsampling to mitigate noise: "To maximally mitigate the adversarial effect, we restrict upsampling to  $2\times$ ." (3.3. Decoder designs)
- Multi-level feature aggregation decoder: "The third design is characterized by multi-level feature aggregation (Figure 1(c))" (3.3. Decoder designs)
- Output stride constraint for memory: "we use output stride 16 in all our models due to GPU memory constrain." (4.1. Experimental setup)

---

## 12. Explicit Limitations and Non-Claims

- Stated limitations or future work: Not specified.
- Explicit statements about what the model does not attempt to do: Not specified.

---

### 13. Constraint Profile (Synthesis)

**Constraint Profile:**
- Domain scope: multiple natural-image segmentation datasets (urban scenes and scene parsing) within a single modality.
- Task structure: single task (semantic segmentation) evaluated on Cityscapes, ADE20K, and PASCAL Context.
- Representation rigidity: fixed 16x16 patch grid, sequence length L = HW/256, fixed output stride 16, and fixed cropping sizes per dataset.
- Model sharing vs specialization: per-dataset training described, with ImageNet pretraining used to initialize the transformer.
- Role of positional encoding: learned absolute positional embeddings added to inputs and interpolated for input size changes.

---

### 14. Final Classification

**Single-task, single-domain.** The paper evaluates semantic segmentation only: "We conduct experiments on three widely-used semantic segmentation benchmark datasets." (4.1. Experimental setup). The datasets are all natural images (e.g., "images with urban scenes" and scene parsing benchmarks), and no cross-domain or multi-task training is claimed. (4.1. Experimental setup)
