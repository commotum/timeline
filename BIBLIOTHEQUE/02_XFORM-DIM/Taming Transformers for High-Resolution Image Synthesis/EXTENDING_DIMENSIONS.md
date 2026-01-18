## 1. Basic Metadata
- Title: Taming Transformers for High-Resolution Image Synthesis. Quote: "Taming Transformers for High-Resolution Image Synthesis" (Title).
- Authors: Patrick Esser; Robin Rombach; Björn Ommer. Quote: "Patrick Esser* Robin Rombach* Björn Ommer Heidelberg Collaboratory for Image Processing, IWR, Heidelberg University, Germany *Both authors contributed equally to this work" (Title).
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary
"We demonstrate how combining the effectiveness of the inductive bias of CNNs with the expressivity of transformers enables them to model and thereby synthesize high-resolution images." (Abstract)

## 3. Tasks Evaluated
- Task name: Unconditional image modeling/synthesis. Task type: Generation. Dataset(s): ImageNet (IN); Restricted ImageNet (RIN); LSUN Churches and Towers (LSUN-CT). Domain: RIN is "a subset of animal classes"; LSUN-CT is churches/towers; ImageNet domain not specified. Quote: "**Results** Tab. 1 reports results for unconditional image modeling on ImageNet (IN) [14], Restricted ImageNet (RIN) [65], consisting of a subset of animal classes from ImageNet, LSUN Churches and Towers (LSUN-CT) [79], and for conditional image modeling of RIN conditioned on depth maps obtained with the approach of [60] (D-RIN) and of landscape images collected from Flickr conditioned on semantic layouts (S-FLCKR) obtained with the approach of [7]." (Sec. 4.1).
- Task name: Semantic image synthesis (segmentation-to-image). Task type: Generation. Dataset(s): ADE20K; S-FLCKR; COCO-Stuff. Domain: "semantic segmentation masks"; S-FLCKR is a "webscraped landscapes dataset"; ADE20K/COCO-Stuff domain not specified. Quote: "- (i): **Semantic image synthesis**, where we condition on semantic segmentation masks of ADE20K [83], a webscraped landscapes dataset (S-FLCKR) and COCO-Stuff [6]." (Sec. 4.2).
- Task name: Structure-to-image (depth/edge-to-image). Task type: Generation. Dataset(s): RIN; IN. Domain: depth/edge conditioning; RIN domain is animal classes; IN domain not specified. Quote: "- (ii): **Structure-to-image**, where we use either depth or edge information to synthesize images from both RIN and IN (see Sec. 4.1). The resulting depth-to-image and edge-to-image translations are visualized in Fig. 4 and Fig. 6." (Sec. 4.2).
- Task name: Pose-guided person generation. Task type: Generation. Dataset(s): DeepFashion. Domain: people/person images. Quote: "4th row: Pose-guided person generation on DeepFashion." (Figure 4 caption).
- Task name: Stochastic superresolution. Task type: Generation (superresolution). Dataset(s): IN (ImageNet). Domain: ImageNet domain not specified. Quote: "2nd row: Stochastic superresolution on IN." (Figure 6 caption).
- Task name: Class-conditional image synthesis. Task type: Generation. Dataset(s): RIN; IN. Domain: class-label conditioned images; RIN is animal classes; IN domain not specified. Quote: "(v): Class-conditional image synthesis: Here, the conditioning information c is a single index describing the class label of interest. Results for the RIN and IN dataset are demonstrated in Fig. 4 and Fig. 8, respectively." (Sec. 4.2).
- Task name: Unconditional face synthesis. Task type: Generation. Dataset(s): CelebA-HQ; FFHQ. Domain: faces. Quote: "the results on *unconditional face synthesis* are shown in Tab. 3." (Sec. 4.4). Quote: "| CelebA-HQ $256 \times 256$ |             | FFHQ $256 \times 256$  |            |  |" (Table 3).
- Task name: Image completion (half completions). Task type: Generation. Dataset(s): ImageNet; S-FLCKR. Domain: ImageNet domain not specified; S-FLCKR is landscapes. Quote: "Top row: Completions from unconditional training on ImageNet." (Figure 4 caption). Quote: "Here, we use our f = 16 S-FLCKR model to obtain high-fidelity image completions of the inputs depicted on the left (half completions)." (Figure 27 caption).

## 4. Domain and Modality Scope
- Single domain? No; multiple image domains are evaluated. Quote: "**Results** Tab. 1 reports results for unconditional image modeling on ImageNet (IN) [14], Restricted ImageNet (RIN) [65], consisting of a subset of animal classes from ImageNet, LSUN Churches and Towers (LSUN-CT) [79], and for conditional image modeling of RIN conditioned on depth maps obtained with the approach of [60] (D-RIN) and of landscape images collected from Flickr conditioned on semantic layouts (S-FLCKR) obtained with the approach of [7]." (Sec. 4.1). Quote: "the results on *unconditional face synthesis* are shown in Tab. 3." (Sec. 4.4). Quote: "4th row: Pose-guided person generation on DeepFashion." (Figure 4 caption).
- Multiple domains within the same modality? Yes; multiple datasets and conditional settings are all image-based. Quote: "We show  $256 \times 256$  synthesis results across different conditioning inputs and datasets, all obtained with the same approach to exploit inductive biases of effective CNN based VQGAN architectures in combination with the expressivity of transformer architectures." (Figure 4 caption).
- Multiple modalities? Not explicitly; conditioning includes class labels and spatial maps. Quote: "Our approach is readily applied to conditional synthesis tasks, where both non-spatial information, such as object classes, and spatial information, such as segmentations, can control the generated image." (Abstract).
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Unconditional image modeling/synthesis | No (per-task training described) | Not specified | Not specified | "For each task, we train a VQGAN with m=4 downsampling blocks" (Sec. 4.1). |
| Semantic image synthesis | Not specified | Not specified | Not specified | "- (i): **Semantic image synthesis**, where we condition on semantic segmentation masks of ADE20K [83], a webscraped landscapes dataset (S-FLCKR) and COCO-Stuff [6]." (Sec. 4.2). |
| Structure-to-image (depth/edge) | Not specified | Not specified | Not specified | "- (ii): **Structure-to-image**, where we use either depth or edge information to synthesize images from both RIN and IN" (Sec. 4.2). |
| Pose-guided person generation | Not specified | Not specified | Not specified | "4th row: Pose-guided person generation on DeepFashion." (Figure 4 caption). |
| Stochastic superresolution | Not specified | Not specified | Not specified | "2nd row: Stochastic superresolution on IN." (Figure 6 caption). |
| Class-conditional image synthesis | Not specified | Not specified | Not specified | "(v): Class-conditional image synthesis: Here, the conditioning information c is a single index describing the class label of interest." (Sec. 4.2). |
| Unconditional face synthesis | Not specified | Not specified | Not specified | "the results on *unconditional face synthesis* are shown in Tab. 3." (Sec. 4.4). |
| Image completion | Not specified | Not specified | Not specified | "Here, we use our f = 16 S-FLCKR model to obtain high-fidelity image completions of the inputs depicted on the left (half completions)." (Figure 27 caption). |

## 6. Input and Representation Constraints
- Images are 2D RGB tensors and encoded into a discrete spatial grid; sequences are length h \cdot w. Quote: "any image  $x \in \mathbb{R}^{H \times W \times 3}$  can be represented by a spatial collection of codebook entries  $z_{\mathbf{q}} \in \mathbb{R}^{h \times w \times n_z}$ , where  $n_z$  is the dimensionality of codes. An equivalent representation is a sequence of  $h \cdot w$  indices which specify the respective entries in the learned codebook." (Sec. 3.1).
- Downsampling factor sets latent grid size. Quote: "reduce images of size  $H \times W$  to  $h = H/2^m \times w = W/2^m$" (Sec. 3.2).
- Fixed training sequence length in many experiments. Quote: "we usually set  $|\mathcal{Z}|=1024$  and train all subsequent transformer models to predict sequences of length  $16 \cdot 16$" (Sec. 4).
- Fixed crop size for transformer inputs during training. Quote: "During training, we always crop images to obtain inputs of size  $16 \times 16$  for the transformer, *i.e.* when modeling images with a factor f in the first stage, we use crops of size  $16f \times 16f$ ." (Sec. 4.3).
- Patch-wise cropping for high-resolution generation. Quote: "we therefore have to work patch-wise and crop images to restrict the length of s to a maximally feasible size during training." (Sec. 3.2).
- Spatial conditioning is also discretized into an index grid. Quote: "If the conditioning information c has spatial extent, we first learn another VQGAN to obtain again an index-based representation  $r \in \{0,\dots,|\mathcal{Z}_c|-1\}^{h_c \times w_c}$" (Sec. 3.2).

## 7. Context Window and Attention Structure
- Maximum sequence length (training):  $16 \cdot 16$  tokens in common settings due to GPT2-medium feasibility. Quote: "train all subsequent transformer models to predict sequences of length  $16 \cdot 16$ , as this is the maximum feasible length to train a GPT2-medium architecture (307 M parameters) [58] on a GPU with 12GB VRAM." (Sec. 4).
- Sequence length is tied to latent grid size and crop size. Quote: "The attention mechanism of the transformer puts limits on the sequence length  $h \cdot w$  of its inputs s." (Sec. 3.2). Quote: "During training, we always crop images to obtain inputs of size  $16 \times 16$  for the transformer" (Sec. 4.3).
- Attention type: global self-attention within the windowed sequence. Quote: "The defining characteristic of the transformer architecture [74] is that it models interactions between its inputs solely through attention [2, 36, 52] which enables them to faithfully handle interactions between inputs regardless of their relative position to one another." (Sec. 2).
- Cost-control mechanisms: latent compression and sliding-window sampling. Quote: "This training procedure significantly reduces the sequence length when unrolling the latent code and thereby enables the application of powerful transformer models." (Sec. 3.1). Quote: "To sample images, we then use the transformer in a sliding-window manner as illustrated in Fig. 3." (Sec. 3.2).

## 8. Positional Encoding (Critical Section)
- Positional encoding mechanism used: Not specified. Evidence: "Our transformer model is identical to the GPT2 architecture [58]" (Sec. B), but no positional encoding details are stated.
- Where it is applied: Not specified.
- Fixed across all experiments / modified per task / ablated: Not specified.

## 9. Positional Encoding as a Variable
- Positional encoding as a core research variable? Not stated; the paper varies sequence ordering instead. Quote: "For the \"classical\" domain of transformer models, NLP, the order of tokens is defined by the language at hand. For images and their discrete representations, in contrast, it is not clear which linear ordering to use. In particular, our sliding-window approach depends on a row-major ordering and we thus investigate the performance of the following five different permutations of the input sequence of codebook indices:" (Sec. F).
- Multiple positional encodings compared: Not specified.
- Claim PE choice is "not critical" or secondary: Not stated.

## 10. Evidence of Constraint Masking
- Model sizes: "we vary the model capacities between 85M and 310M parameters" (Sec. 4.1). Quote: "train all subsequent transformer models to predict sequences of length  $16 \cdot 16$ , as this is the maximum feasible length to train a GPT2-medium architecture (307 M parameters) [58] on a GPU with 12GB VRAM." (Sec. 4).
- Dataset size(s): Not quantified; the paper notes data size limits in faces. Quote: "the bottleneck for our approach on face synthesis is given by the dataset size" (Sec. E).
- Performance gains attributed to architecture/representation rather than data scale. Quote: "This training procedure significantly reduces the sequence length when unrolling the latent code and thereby enables the application of powerful transformer models." (Sec. 3.1). Quote: "Only our full setting of f=16 can synthesize high-fidelity samples." (Sec. 4.3). Quote: "We observe improvements of 18.63% for FIDs and  $14.08\times$  faster sampling of images." (Sec. 4.3).

## 11. Architectural Workarounds
- Two-stage CNN + transformer with discrete codebook. Quote: "We show how to (i) use CNNs to learn a contextrich vocabulary of image constituents, and in turn (ii) utilize transformers to efficiently model their composition within high-resolution images." (Abstract).
- Discrete codebook + patch-based discriminator for compression. Quote: "A discrete codebook provides the interface between these architectures and a patch-based discriminator enables strong compression while retaining high perceptual quality." (Figure 2 caption).
- Downsampling to reduce latent grid size. Quote: "reduce images of size  $H \times W$  to  $h = H/2^m \times w = W/2^m$" (Sec. 3.2).
- Patch-wise training and sliding-window sampling. Quote: "To generate images in the megapixel regime, we therefore have to work patch-wise and crop images to restrict the length of s to a maximally feasible size during training. To sample images, we then use the transformer in a sliding-window manner as illustrated in Fig. 3." (Sec. 3.2).
- Extra attention at lowest resolution in VQGAN. Quote: "To aggregate context from everywhere, we apply a single attention layer on the lowest resolution." (Sec. 3.1).

## 12. Explicit Limitations and Non-Claims
- Reconstruction degrades beyond a dataset-dependent compression level. Quote: "we observe degradation of the reconstruction quality beyond a critical value of m, which depends on the considered dataset." (Sec. 3.2).
- High-resolution sampling assumes spatial invariance or spatial conditioning. Quote: "Our VQGAN ensures that the available context is still sufficient to faithfully model images, as long as either the statistics of the dataset are approximately spatially invariant or spatial conditioning information is available." (Sec. 3.2).
- Dataset size limits face synthesis. Quote: "the bottleneck for our approach on face synthesis is given by the dataset size" (Sec. E).
- Evaluation limitations for overfitting. Quote: "it is not clear if early-stopping based on it is optimal if one is mainly interested in the quality of samples. To address this and the evaluation of GANs, new metrics will be required" (Sec. E).
- Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)
> **Constraint Profile:**
> - Domain scope: multiple image domains (ImageNet/RIN/LSUN-CT, faces, landscapes, DeepFashion) within a single visual modality.
> - Task structure: multiple conditional and unconditional image synthesis tasks (semantic synthesis, structure-to-image, pose-guided, superresolution, class-conditional, completion).
> - Representation rigidity: images mapped to fixed-size latent grids with downsampling; common training crop is  $16 \times 16$  and sequence length  $16 \cdot 16$ .
> - Model sharing vs specialization: per-task training is described in Sec. 4.1; no explicit joint multi-task training or shared transformer weights.
> - Role of positional encoding: not specified; only sequence ordering permutations are analyzed.

### 14. Final Classification
**Multi-task, multi-domain (constrained).** The paper evaluates multiple synthesis tasks ("Semantic image synthesis," "Structure-to-image," "Pose-guided person generation," "Stochastic superresolution," and class-conditional generation) across multiple datasets (ImageNet/RIN/LSUN-CT, Faces, DeepFashion, S-FLCKR) within the image modality. The setup remains constrained by fixed latent-grid representations and bounded sequence lengths (e.g., "predict sequences of length  $16 \cdot 16$") rather than open-ended multi-domain transfer.
