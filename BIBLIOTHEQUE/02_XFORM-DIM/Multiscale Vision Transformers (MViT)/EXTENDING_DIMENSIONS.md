## 1. Basic Metadata

- Title: "Multiscale Vision Transformers" (Title block)
- Authors: "Haoqi Fan *, 1 Bo Xiong *, 1 Karttikeya Mangalam *, 1, 2
Yanghao Li *, 1 Zhicheng Yan 1 Jitendra Malik 1, 2 Christoph Feichtenhofer *, 1" (Title block)
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

The paper presents "Multiscale Vision Transformers (MViT) for video and image recognition" that connect the "multiscale feature hierarchies with transformer models" to model dense visual signals. (Abstract)

## 3. Tasks Evaluated

- Task name: Video recognition (classification) on Kinetics-400/600.
  - Task type: Classification.
  - Dataset(s) used: Kinetics-400 (K400), Kinetics-600.
  - Domain: video (natural videos).
  - Quotes: "Our focus in this paper is video recognition, and we design and evaluate MViT for video tasks (Kinetics [59, 10], Charades [86], SSv2 [38] and AVA [39])." (Introduction) "We use Kinetics-400 [59] (K400) (~240k training videos in 400 classes) and Kinetics-600 [11]." (Section 4. Experiments: Video Recognition)

- Task name: Video recognition (temporal modeling) on Something-Something-v2.
  - Task type: Classification.
  - Dataset(s) used: Something-Something-v2 (SSv2).
  - Domain: video (object interactions).
  - Quotes: "We further assess transfer learning performance for on Something-Something-v2 [38], Charades [86], and AVA [39]." (Section 4. Experiments: Video Recognition) "Something-Something-v2 (SSv2) [38] is a dataset with videos containing object interactions, which is known as a 'temporal modeling' task." (Section 4.1)

- Task name: Video recognition (long-range activities) on Charades.
  - Task type: Classification.
  - Dataset(s) used: Charades.
  - Domain: video (longer range activities).
  - Quotes: "Charades [86] is a dataset with longer range activities. We validate our model in Table 7." (Section 4.1)

- Task name: Spatiotemporal action localization on AVA.
  - Task type: Detection.
  - Dataset(s) used: AVA.
  - Domain: video (spatiotemporal localization of human actions).
  - Quotes: "AVA [39] is a dataset with for spatiotemporal-localization of human actions. We validate our MViT on this detection task." (Section 4.1)

- Task name: Image recognition on ImageNet-1K.
  - Task type: Classification.
  - Dataset(s) used: ImageNet-1K.
  - Domain: natural images.
  - Quotes: "We apply our video models on static image recognition by using them with single frame, T=1, on ImageNet-1K [22]." (Section 5. Experiments: Image Recognition) "Then we train and validate them ("from scratch") on ImageNet." (Section 5.1)

## 4. Domain and Modality Scope

- Evaluation spans video and image recognition within vision: "We present Multiscale Vision Transformers (MViT) for video and image recognition" (Abstract) and "We apply our video models on static image recognition by using them with single frame, T=1, on ImageNet-1K [22]." (Section 5. Experiments: Image Recognition)
- Single domain vs multiple domains: Multiple domains within the same modality (vision: video + images); multiple modalities are not claimed.
- Domain generalization or cross-domain transfer: Transfer learning across video datasets is mentioned ("We further assess transfer learning performance for on Something-Something-v2 [38], Charades [86], and AVA [39]." (Section 4. Experiments: Video Recognition)); cross-domain or cross-modality generalization is not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Kinetics-400/600 video recognition | No; trained from scratch per dataset. | No; from scratch. | Yes; linear classifier head on class embedding. | "By default, all models are trained *from random initialization* ("*from scratch*") on Kinetics, *without* using ImageNet [22] or other pre-training." (Section 4. Experiments: Video Recognition) "We train MViT from-scratch, without any pre-training." (Section 4.1) "The resulting sequence after N consecutive blocks is layer-normalized and the class embedding is extracted and passed through a linear layer to predict the desired output (e.g. class)." (Section 3.2) |
| Something-Something-v2 video recognition | Yes; K600 pre-trained weights used. | Not explicitly stated; transfer learning implied. | Yes; linear classifier head on class embedding. | "We further assess transfer learning performance for on Something-Something-v2 [38], Charades [86], and AVA [39]." (Section 4. Experiments: Video Recognition) "MViT-B-24 achieves 68.7% using our K600 pre-trained model of above." (Section 4.1) "The resulting sequence after N consecutive blocks is layer-normalized and the class embedding is extracted and passed through a linear layer to predict the desired output (e.g. class)." (Section 3.2) |
| Charades video recognition | Yes; K600 pre-trained weights used. | Not explicitly stated; transfer learning implied. | Yes; linear classifier head on class embedding. | "the performance of **MViT**-B is further improved by increasing the number of input frames and **MViT**-B layers and using K600 pre-trained models." (Section 4.1) "The resulting sequence after N consecutive blocks is layer-normalized and the class embedding is extracted and passed through a linear layer to predict the desired output (e.g. class)." (Section 3.2) |
| AVA spatiotemporal action localization | Pre-training used; exact sharing not specified. | Not specified. | Yes; detection architecture referenced. | "We observe that MViT-B can be competitive to SlowFast and X3D using the same pre-training and testing strategy." (Section 4.1) "AVA [39] is a dataset with for spatiotemporal-localization of human actions. We validate our MViT on this detection task. Details about the detection architecture of MViT can be found in §D.2." (Section 4.1) |
| ImageNet-1K image recognition | No; trained from scratch on ImageNet. | No; from scratch. | Yes; linear classifier head on class embedding. | "Then we train and validate them ("from scratch") on ImageNet." (Section 5.1) "The resulting sequence after N consecutive blocks is layer-normalized and the class embedding is extracted and passed through a linear layer to predict the desired output (e.g. class)." (Section 3.2) |

## 6. Input and Representation Constraints

- Input is explicitly 3D space-time for video: "input video of resolution  $T \times H \times W$" (Section 3.2) and image models are single-frame: "single frame, T=1" (Section 5).
- Fixed patch/cube sizes and strides are specified: "non-overlapping patches of size  $1 \times 16 \times 16$  each" (Section 3.2); "cube_1, projects *dense* space-time cubes (of shape  $c_T \times c_H \times c_W$ ) to D channels to reduce spatio-temporal resolution to  $\frac{T}{s_T} \times \frac{H}{4} \times \frac{W}{4}$ ." (Table 2); "MViT-B initially projects the input to a channel dimension of D = 96 with overlapping space-time cubes of shape  $3\times7\times7$ ." (Section 3.3)
- Fixed token counts for given input resolution are given: "The sequence length (spacetime resolution + class token) is  $8 \cdot 14 \cdot 14 + 1 = 1569$ ." (Section 3.3) and "The resulting sequence of length 8\*56\*56+1 = 25089 is reduced by a factor of 4 for each additional stage, to a final sequence length of 8\*7\*7+1=393 at scale<sub>4</sub>." (Section 3.3)
- Example fixed input resolution/crop for experiments: "With an  $8\times224\times224$  input the resolution is fixed to  $768 \times 8 \times 14 \times 14$  throughout all layers." (Section 3.3) and inference uses "a  $224 \times 224$  center crop" (Section 4. Experiments: Video Recognition).
- Padding is specified for pooling: "By default we use *overlapping* kernels  $\mathbf{k}$  with *shape-preserving* padding  $\mathbf{p}$" (Section 3.1).
- A class token is appended: "A learnable class embedding is appended to the projected image patches." (Section 3.2).

## 7. Context Window and Attention Structure

- Maximum sequence length (example): "The resulting sequence of length 8\*56\*56+1 = 25089 is reduced by a factor of 4 for each additional stage, to a final sequence length of 8\*7\*7+1=393" (Section 3.3); for ViT-B, "The sequence length (spacetime resolution + class token) is  $8 \cdot 14 \cdot 14 + 1 = 1569$ ." (Section 3.3).
- Sequence length fixed or variable: It is defined by input resolution and pooling: "input video of resolution  $T \times H \times W$" (Section 3.2) and explicit sequence-length computations are given for specific inputs (Section 3.3).
- Attention type: Hierarchical + pooled attention: "Multiscale Transformers have several channel-resolution 'scale' stages. Starting from the input resolution and a small channel dimension, the stages hierarchically expand the channel capacity while reducing the spatial resolution." (Abstract) and "MHPA *pools* the sequence of latent tensors to reduce the sequence length (resolution) of the attended input." (Section 3.1).
- Mechanisms to manage compute: "Since attention computation scales quadratically w.r.t. the sequence length, pooling the key, query and value tensors has dramatic benefits on the fundamental compute and memory requirements" (Section 3.1) and "only the first pooling attention operator of each stage operates at non-degenerate query stride  $s^Q > 1$ , with all other operators constrained to  $\mathbf{s}^Q \equiv (1, 1, 1)$ ." (Section 3.2).

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: "a positional embedding  $\mathbf{E} \in \mathbb{R}^{L \times D}$  is added to each element of the projected sequence of length L" (Section 3.2), implying a learnable absolute embedding.
- Where applied: Added to the input sequence only: "positional embedding ... is added to each element of the projected sequence" (Section 3.2).
- Space/time variants and defaults: "we ablate using (i) none, (ii) space-only, (iii) joint space-time, and (iv) a separate space and time (our default), positional embeddings." (Table 11, Section 4.2)
- Temporal positional embeddings are used and adjusted: "All models are trained without any shuffling and have temporal embeddings." (Section 4.2) and "after interpolating the temporal positional embedding" (Section 4.1).

## 9. Positional Encoding as a Variable

- Treated as a research variable in ablations rather than a fixed assumption: "we ablate using (i) none, (ii) space-only, (iii) joint space-time, and (iv) a separate space and time (our default), positional embeddings." (Table 11, Section 4.2)
- Multiple positional encodings compared: yes, the ablation explicitly compares four variants (Table 11, Section 4.2).
- PE "not critical" claim: Not claimed.

## 10. Evidence of Constraint Masking

- Architectural hierarchy vs scaling data/compute: "We evaluate this fundamental architectural prior ... where it outperforms concurrent vision transformers that rely on large scale external pre-training and are 5-10× more costly in computation and parameters." (Abstract)
- Model size vs performance: "Our base model, MViT-B provides 78.4%, a +9.9% accuracy boost over ViT-B under *identical settings*, while having  $2.6\times/2.4\times$  fewer FLOPs/parameters." (Section 4.1)
- Dataset size and training data scale: "Kinetics-400 [59] (K400) (~240k training videos in 400 classes)" (Section 4) and "By default, all models are trained *from random initialization* ("*from scratch*") on Kinetics, *without* using ImageNet [22] or other pre-training." (Section 4)
- Model scaling is explored explicitly: "We further train a deeper 24-layer model with longer sampling, MViT-B-24, 32×3, to investigate model scale on this larger dataset." (Section 4.1)

## 11. Architectural Workarounds

- Multiscale hierarchy to reduce resolution while increasing channels: "Multiscale Transformers have several channel-resolution 'scale' stages... [that] hierarchically expand the channel capacity while reducing the spatial resolution." (Abstract)
- Pooling attention to reduce sequence length: "MHPA *pools* the sequence of latent tensors to reduce the sequence length (resolution) of the attended input." (Section 3.1)
- Query pooling only at stage start; K/V pooling elsewhere: "only the first pooling attention operator of each stage operates at non-degenerate query stride  $s^Q > 1$ , with all other operators constrained to  $\mathbf{s}^Q \equiv (1, 1, 1)$ ." (Section 3.2) and "We employ K, V pooling in all MHPA blocks" (Section 3.3).
- Cube/patch projection to reduce resolution: "cube_1, projects *dense* space-time cubes ... to reduce spatio-temporal resolution" (Table 2) and "non-overlapping patches of size  $1 \times 16 \times 16$  each" (Section 3.2).
- Skip-connection adjustments for scale changes: "we pool the skip connection to adapt to the dimension mismatch between its two ends." (Section 3.2) and "we employ an extra linear layer that operates on the layer-normalized output of our MHPA operation." (Section 3.2).
- Class token excluded from pooling: "Note that all pooling operations, and hence the resolution downsampling, is performed only on the data sequence without involving the processed class token embedding." (Section 3.3)

## 12. Explicit Limitations and Non-Claims

- Scope statement: "Our focus in this paper is video recognition, and we design and evaluate MViT for video tasks (Kinetics [59, 10], Charades [86], SSv2 [38] and AVA [39])." (Introduction)
- Limitation/future work on training recipe transfer: "The results suggest that SlowFast models do not benefit from the MViT recipe directly and more studies are required to understand the effect of applying our training-from-scratch recipe to ConvNets, as it seems higher capacity ConvNets (R101) perform worse when using our recipe." (Appendix A.1)

## 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Visual-only; evaluated on video and image recognition datasets, not multiple modalities.
> - Task structure: Separate evaluations for video classification/recognition and image classification, plus a video detection task (AVA); no joint multi-task training described.
> - Representation rigidity: Fixed patch/cube sizes, fixed crops (e.g., 224×224), and sequence lengths tied to T×H×W with a class token.
> - Model sharing vs specialization: Kinetics and ImageNet trained from scratch; transfer learning with K600/K400 pre-training for some video datasets.
> - Role of positional encoding: Learnable positional embeddings added to inputs; ablations compare none/space-only/joint/separate and temporal embedding interpolation.

## 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple tasks within the visual modality, including "video and image recognition" (Abstract) and a video detection task on AVA, but all remain within vision. It reports separate training or transfer learning per dataset (e.g., "By default, all models are trained *from random initialization* ("*from scratch*") on Kinetics" (Section 4) and "Then we train and validate them ("from scratch") on ImageNet." (Section 5.1)), rather than joint multi-task training.
