## 1. Basic Metadata

- Title: "Perceiver: General Perception with Iterative Attention" (Title)
- Authors: "Andrew Jaegle <sup>1</sup> Felix Gimeno <sup>1</sup> Andrew Brock <sup>1</sup> Andrew Zisserman <sup>1</sup> Oriol Vinyals <sup>1</sup> Joao Carreira <sup>1</sup>" (Title)
- Year: "Proceedings of the 38<sup>th</sup> International Conference on Machine Learning, PMLR 139, 2021." (Introduction)
- Venue (conference/journal/arXiv): "Proceedings of the 38<sup>th</sup> International Conference on Machine Learning, PMLR 139, 2021." (Introduction)


## 2. One-Sentence Contribution Summary

"In this paper we introduce the Perceiver – a model that builds upon Transformers and hence makes few architectural assumptions about the relationship between its inputs, but that also scales to hundreds of thousands of inputs, like ConvNets." (Abstract)


## 3. Tasks Evaluated

General scope quote: "We show that this architecture is competitive with or outperforms strong, specialized models on classification tasks across various modalities: images, point clouds, audio, video, and video+audio." (Abstract)

- Task name: ImageNet single-image classification
  - Task type: Classification
  - Dataset(s) used: ImageNet (ILSVRC 2012)
  - Domain: natural images
  - Quotes: "First, we consider the task of single-image classification using the ILSVRC 2012 split of the ImageNet dataset (Deng et al., 2009)." (Section 4.1. Images - ImageNet) "Each image on ImageNet has a single label so we use softmax outputs and a cross-entropy loss to train for the classification task." (Section 4.1. Images - ImageNet)

- Task name: AudioSet audio event classification (audio, video, audio+video inputs)
  - Task type: Classification (multi-label)
  - Dataset(s) used: AudioSet
  - Domain: audio and video
  - Quotes: "We experimented with audio event classification in video using AudioSet (Gemmeke et al., 2017), a large dataset with 1.7M 10s long training videos and 527 classes." (Section 4.2. Audio and video - AudioSet) "We evaluate the Perceiver using audio (using either the raw audio waveform or mel spectrogram), video, and audio + video as inputs." (Section 4.2. Audio and video - AudioSet) "Videos may have multiple labels so we use a sigmoid cross entropy loss and evaluate using mean average precision (mAP)." (Section 4.2. Audio and video - AudioSet)

- Task name: ModelNet40 point cloud classification
  - Task type: Classification
  - Dataset(s) used: ModelNet40
  - Domain: 3D point clouds
  - Quotes: "ModelNet40 (Wu et al., 2015) is a dataset of point clouds derived from 3D triangular meshes spanning 40 object categories. The task is to predict the class of each object, given the coordinates of  $\sim$  2000 points in 3D space." (Section 4.3. Point clouds - ModelNet40)


## 4. Domain and Modality Scope

- Evaluation spans multiple modalities/domains: "classification tasks across various modalities: images, point clouds, audio, video, and video+audio." (Abstract)
- Multiple domains across modalities: "We train the Perceiver architecture on images from ImageNet (Deng et al., 2009) (left), video and audio from AudioSet (Gemmeke et al., 2017) (considered both multi- and uni-modally) (center), and 3D point clouds from ModelNet40 (Wu et al., 2015) (right)." (Figure 2 caption)
- Domain generalization or cross-domain transfer: Not claimed.


## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| ImageNet single-image classification | Not stated (separate training per dataset described) | Not stated | Yes (task-specific output/loss) | "We train our model using images sampled by Inception-style preprocessing (Szegedy et al., 2015), including standard  $224 \times 224$  pixel crops." (Section 4.1. Images - ImageNet) "Each image on ImageNet has a single label so we use softmax outputs and a cross-entropy loss to train for the classification task." (Section 4.1. Images - ImageNet) |
| AudioSet audio event classification | Not stated (separate training per dataset described) | Not stated | Yes (task-specific output/loss) | "We train models for 100 epochs." (Section 4.2. Audio and video - AudioSet) "Videos may have multiple labels so we use a sigmoid cross entropy loss and evaluate using mean average precision (mAP)." (Section 4.2. Audio and video - AudioSet) |
| ModelNet40 point cloud classification | Not stated (separate training per dataset described) | Not stated | Not specified | "ModelNet40 (Wu et al., 2015) is a dataset of point clouds derived from 3D triangular meshes spanning 40 object categories. The task is to predict the class of each object, given the coordinates of  $\sim$  2000 points in 3D space." (Section 4.3. Point clouds - ModelNet40) "We used an architecture with 2 cross-attentions and 6 self-attention layers for each block and otherwise used the same architectural settings as ImageNet." (Section 4.3. Point clouds - ModelNet40) |


## 6. Input and Representation Constraints

- ImageNet resolution/token count: "ImageNet images at resolution 224 have 50,176 pixels" (Section 3.1. The Perceiver architecture) and "standard  $224 \times 224$  pixel crops" (Section 4.1. Images - ImageNet).
- Audio windowing and tokenization: "audio sampled at 48Khz resulting in 61,440 audio samples over 1.28s of video" (Section 4.2. Audio and video - AudioSet); "we divide the raw signal into segments of 128 elements, for a total of 480 128-d vectors" (Section 4.2. Audio and video - AudioSet); "a mel spectrogram resulting in 4800 inputs to the Perceiver, once flattened" (Section 4.2. Audio and video - AudioSet).
- Video clip length, resolution, and patching: "We sample 32-frame clips (1.28s at 25fps)" (Section 4.2. Audio and video - AudioSet); "A full 32 frame clip at 224x224 resolution has more than 2 million pixels" (Section 4.2. Audio and video - AudioSet); "tiny space-time patches with dimensions 2x8x8, resulting in a total of 12,544 inputs to the Perceiver" (Section 4.2. Audio and video - AudioSet).
- Multimodal channel constraint: "Since modalities are fused at input, audio and video inputs need to have the same number of channels. We achieve this by concatenating a learned, modality-specific encoding to each input." (Section 4.2. Audio and video - AudioSet)
- Point cloud size and dimensionality: "the coordinates of  $\sim$  2000 points in 3D space" (Section 4.3. Point clouds - ModelNet40).
- Latent bottleneck size: "we use values of  $N \leq 1024$" and "we use 512 latents on ImageNet" (Section 3.1. The Perceiver architecture).
- Positional dimensionality assumptions: "preserving 1D temporal or 2D spatial structure for audio or images, respectively, or 3D spatiotemporal structure for videos" (Section 3.2. Position encodings).
- Padding/resizing requirements: Not specified; cropping is used ("standard  $224 \times 224$  pixel crops"; Section 4.1. Images - ImageNet).


## 7. Context Window and Attention Structure

- Maximum sequence length (examples given): "ImageNet images at resolution 224 have 50,176 pixels" (Section 3.1. The Perceiver architecture); "A full 32 frame clip at 224x224 resolution has more than 2 million pixels" (Section 4.2. Audio and video - AudioSet); "audio sampled at 48Khz resulting in 61,440 audio samples over 1.28s of video" (Section 4.2. Audio and video - AudioSet); "the coordinates of  $\sim$  2000 points in 3D space" (Section 4.3. Point clouds - ModelNet40).
- Fixed or variable length: "The size of the byte array is determined by the input data and is generally large" (Section 3.1. The Perceiver architecture), with fixed preprocessing per dataset (e.g., "standard  $224 \times 224$  pixel crops"; Section 4.1. Images - ImageNet).
- Attention type: Global/non-causal attention: "All attention modules in the Perceiver are non-causal: we use no masks." (Section 3.1. The Perceiver architecture) and "The Perceiver uses a cross-attention module to project an high-dimensional input byte array to a fixed-dimensional latent bottleneck (the number of input indices M is much larger than the number of latent indices N) before processing it using a deep stack of Transformer-style self-attention blocks in the latent space." (Figure 1 caption).
- Computational cost mechanisms: "introduce a small set of latent units that forms an attention bottleneck" (Section 1. Introduction); "The resulting cross-attention operation has complexity  $\mathcal{O}(MN)$" (Section 3.1. The Perceiver architecture); "This results in an architecture with complexity  $\mathcal{O}(MN+LN^2)$" (Section 3.1. The Perceiver architecture); "we can increase the parameter efficiency of the model by sharing weights between the corresponding blocks of each latent Transformer and/or between cross-attend modules" (Section 3.1. The Perceiver architecture).


## 8. Positional Encoding (Critical Section)

- Mechanism: "Fourier feature position encodings" (Section 3.2. Position encodings); "By replacing these features with a fully learned, 128-dimensional position encoding" (Section 4.1. Images - ImageNet); and for multimodal inputs, "concatenating a learned, modality-specific encoding to each input" where "This encoding doubles as a modality-specific position encoding" (Section 4.2. Audio and video - AudioSet).
- Where applied: "tagging *position encodings* onto the input features" (Section 3.2. Position encodings) and "concatenate the position and input features before passing them into the Perceiver" (Section 3.2. Position encodings). The latent array is also positional: "The latent array itself is initialized using a learned position encoding" (Section 3.1. The Perceiver architecture).
- Fixed vs modified across experiments: "position encodings generated with 64 bands and a maximum resolution of 224 pixels" (Section 4.1. Images - ImageNet); "By replacing these features with a fully learned, 128-dimensional position encoding" (Section 4.1. Images - ImageNet); "We used a higher maximum frequency than for image data to account for the irregular sampling structure of point clouds - we used a max frequency of  $1120 (10 \times$  the value used on ImageNet)." (Section 4.3. Point clouds - ModelNet40); "we found that generating position coordinates using cropped data rather than on the raw data was important to prevent excessive overfitting." (Appendix D. Position encodings and Fourier features)


## 9. Positional Encoding as a Variable

- Position encoding is treated as a variable rather than fixed: "By replacing these features with a fully learned, 128-dimensional position encoding, we can evaluate the performance of a Perceiver with no knowledge of the spatial structure of the inputs." (Section 4.1. Images - ImageNet)
- Multiple positional encodings are compared or adjusted: "In early experiments, we tried using image coordinates rather than crop coordinates as the basis of the position encoding, but we found that this led to model overfitting." (Section 4.1. Images - ImageNet) and "These experiments suggest that standard and relatively small values for the initialization scale are best (values  $\geq 1$  may lead to instability), and generally suggest that a higher number of Fourier frequency bands and a higher maximum resolution (up to Nyquist) improve performance." (Appendix B. Ablations)
- Claim that PE choice is not critical or secondary: Not claimed.


## 10. Evidence of Constraint Masking

- Model scale: "our best ImageNet results use a network with 48 latent Transformer blocks" (Section 3.1. The Perceiver architecture); "we use values of  $N \leq 1024$" (Section 3.1. The Perceiver architecture); "The resulting model has  $\sim 45$  million parameters" (Section 4.1. Images - ImageNet).
- Dataset scale: "AudioSet (Gemmeke et al., 2017), a large dataset with 1.7M 10s long training videos and 527 classes." (Section 4.2. Audio and video - AudioSet) and "ModelNet is small compared to other datasets used in our experiments: it has 9,843 training examples and 2,468 testing examples." (Section 4.3. Point clouds - ModelNet40)
- Scaling vs structure attribution: "As discussed in Sec. 3, it is this decoupling and not merely linear scaling that allows us to build very deep architectures, which appear to be essential for good performance on challenging tasks in a range of domains." (Section 2. Related Work) and "These results suggest that increasing the size of the model tends to produce better results." (Appendix B. Ablations)
- Training tricks noted: "we augment all images using RandAugment" (Section 4.1. Images - ImageNet); "For spectrograms we use also specaugment" (Section 4.2. Audio and video - AudioSet); "video dropout – entirely zeroing out the video stream during training" (Section 4.2. Audio and video - AudioSet)
- Primary attribution to scaling data vs architecture: Not explicitly stated beyond the statements above.


## 11. Architectural Workarounds

- Attention bottleneck to manage scale: "introduce a small set of latent units that forms an attention bottleneck through which the inputs must pass" (Section 1. Introduction) and "The resulting cross-attention operation has complexity  $\mathcal{O}(MN)$" (Section 3.1. The Perceiver architecture).
- Iterative cross-attention: "The Perceiver iteratively attends to the input byte array by alternating cross-attention and latent self-attention blocks." (Figure 1 caption)
- Decoupling depth from input size: "This results in an architecture with complexity  $\mathcal{O}(MN+LN^2)$ , and this is key: by decoupling the input size and the depth, we can add additional Transformer layers at a cost that's independent of the input size." (Section 3.1. The Perceiver architecture)
- Weight sharing for parameter efficiency: "we can increase the parameter efficiency of the model by sharing weights between the corresponding blocks of each latent Transformer and/or between cross-attend modules" (Section 3.1. The Perceiver architecture).
- Token/patch reduction for large inputs: "we divide the raw signal into segments of 128 elements" (Section 4.2. Audio and video - AudioSet) and "tiny space-time patches with dimensions 2x8x8, resulting in a total of 12,544 inputs to the Perceiver" (Section 4.2. Audio and video - AudioSet).


## 12. Explicit Limitations and Non-Claims

- Overfitting risk: "With great flexibility comes great overfitting, and many of our design decisions were made to mitigate this." (Section 5. Discussion)
- Pretraining is future work: "In future work, we would like to pre-train our image classification model on very large scale data" (Section 5. Discussion)
- Modality-agnostic learning not achieved: "While we reduced the amount of modality-specific prior knowledge in the model, we still employ modality-specific augmentation and position encoding. End-to-end modalityagnostic learning remains an interesting research direction." (Section 5. Discussion)
- Multimodal results below SOTA: "Audio+video fusion leads to solid improvements over single modalities (and outperforms specialized fusion optimization approaches (Wang et al., 2020c)) but is still lower than the state-of-the-art approach that uses separate models with late fusion (Fayek & Kumar, 2020). We will investigate this in future work." (Section 4.2. Audio and video - AudioSet)
- Training instability in a setting: "We found that sharing the initial cross-attention with subsequent cross-attends led to instability in training" (Section 4.1. Images - ImageNet)
- Explicit non-claims about open-world learning or unrestrained multitask learning: Not stated.


### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: Multiple modalities (images, audio, video, point clouds) evaluated on separate benchmarks.
> – Task structure: Supervised classification only (single-label ImageNet; multi-label AudioSet; point-cloud classification).
> – Representation rigidity: Fixed preprocessing (224x224 crops, 32-frame clips, fixed patch/segment sizes, ~2000 points) and a fixed latent bottleneck (N <= 1024).
> – Model sharing vs specialization: Same architecture family, but trained per dataset with task-specific outputs/losses; no joint multi-task training reported.
> – Role of positional encoding: Explicit Fourier-feature or learned absolute encodings, tuned per modality and ablated/compared.


### 14. Final Classification

**Multi-task, multi-domain (constrained).** The paper evaluates "classification tasks across various modalities: images, point clouds, audio, video, and video+audio" (Abstract) and trains on ImageNet, AudioSet, and ModelNet40 (Figure 2 caption), so the scope is multi-domain. Each dataset uses task-specific outputs/losses (e.g., "softmax outputs and a cross-entropy loss" for ImageNet; "sigmoid cross entropy loss" for AudioSet), and no cross-domain transfer is claimed, so the setting is constrained to separate supervised tasks rather than unrestrained multi-task learning.
