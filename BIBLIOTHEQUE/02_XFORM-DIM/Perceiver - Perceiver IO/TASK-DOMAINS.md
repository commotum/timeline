# Perceiver: General Perception with Iterative Attention (2021)
Source: Perceiver - Perceiver IO.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Single-image classification (ImageNet) | images (pixels) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | class label (single) | 0D (inferred) | Fixed (inferred) |
| Audio event classification (AudioSet) | audio waveform; mel spectrogram; video frames; audio+video | 1D (t); 2D (x, y); 3D (x, y, t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | multi-label class labels (audio events) | 0D (inferred) | Fixed (inferred) |
| Point cloud classification (ModelNet40) | point clouds (3D point coordinates) | 3D (x, y, z) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | class label (object category) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates the Perceiver on supervised classification across images, audio/video (including audio+video), and 3D point clouds. Inputs span 2D images, 1D audio waveforms/2D spectrograms, 3D video clips, and 3D point coordinates, with fixed-size or bounded preprocessing in the reported setups. Attention is applied over fixed input arrays and the model constructs a latent bottleneck, so attention/state dynamics are treated as static/constructed (inferred).

## Evidence
### Task: Single-image classification (ImageNet)
- "First, we consider the task of single-image classification using the ILSVRC 2012 split of the ImageNet dataset (Deng et al., 2009)." (Section 4.1. Images - ImageNet)
- "Each image on ImageNet has a single label so we use softmax outputs and a cross-entropy loss to train for the classification task." (Section 4.1. Images - ImageNet)
- "including standard  $224 \times 224$  pixel crops." (Section 4.1. Images - ImageNet)
- "project an high-dimensional input byte array to a fixed-dimensional latent bottleneck" (Figure 1 caption)
- "All attention modules in the Perceiver are non-causal: we use no masks." (Section 3.1. The Perceiver architecture)
- Inference: In Dimension 2D (x, y) and Fixed input inferred from the 224x224 pixel crops; output 0D/fixed from the single-label classification; attention Static and state Constructed inferred from non-causal full attention and the latent bottleneck.

### Task: Audio event classification (AudioSet)
- "We experimented with audio event classification in video using AudioSet (Gemmeke et al., 2017)" (Section 4.2. Audio and video - AudioSet)
- "We evaluate the Perceiver using audio (using either the raw audio waveform or mel spectrogram), video, and audio + video as inputs." (Section 4.2. Audio and video - AudioSet)
- "Videos may have multiple labels so we use a sigmoid cross entropy loss" (Section 4.2. Audio and video - AudioSet)
- "1.7M 10s long training videos and 527 classes." (Section 4.2. Audio and video - AudioSet)
- "We sample 32-frame clips (1.28s at 25fps) in training" (Section 4.2. Audio and video - AudioSet)
- "We use audio sampled at 48Khz resulting in 61,440 audio samples over 1.28s of video." (Section 4.2. Audio and video - AudioSet)
- "project an high-dimensional input byte array to a fixed-dimensional latent bottleneck" (Figure 1 caption)
- "All attention modules in the Perceiver are non-causal: we use no masks." (Section 3.1. The Perceiver architecture)
- Inference: In Dimension includes 1D (t), 2D (x, y), and 3D (x, y, t) from raw audio, mel spectrograms, and video clips; Fixed input length inferred from 32-frame/1.28s sampling; output 0D/fixed from the labeled-class setup; attention Static and state Constructed inferred from non-causal full attention and the latent bottleneck.

### Task: Point cloud classification (ModelNet40)
- "ModelNet40 (Wu et al., 2015) is a dataset of point clouds derived from 3D triangular meshes spanning 40 object categories." (Section 4.3. Point clouds - ModelNet40)
- "The task is to predict the class of each object, given the coordinates of  $\sim$  2000 points in 3D space." (Section 4.3. Point clouds - ModelNet40)
- "project an high-dimensional input byte array to a fixed-dimensional latent bottleneck" (Figure 1 caption)
- "All attention modules in the Perceiver are non-causal: we use no masks." (Section 3.1. The Perceiver architecture)
- Inference: In Dimension 3D (x, y, z) inferred from "3D space"; input dynamics marked Capped from "~ 2000 points"; output 0D/fixed from "40 object categories"; attention Static and state Constructed inferred from non-causal full attention and the latent bottleneck.
