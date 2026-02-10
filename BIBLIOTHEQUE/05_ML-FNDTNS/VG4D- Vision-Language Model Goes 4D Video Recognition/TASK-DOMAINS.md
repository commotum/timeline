# VG4D: Vision-Language Model Goes 4D Video Recognition (Not specified in the paper.)
Source: VG4D- Vision-Language Model Goes 4D Video Recognition.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cross-modal contrastive alignment (representation learning) | 4D point cloud video; RGB video; action-category text descriptions | 4D (x, y, z, t); 3D (x, y, z) or (x, y, t) (inferred); 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Aligned modality embeddings and contrastive similarity scores (inferred) | 0D (inferred) | Fixed (inferred) |
| 4D point cloud action recognition (classification) | 4D point cloud video | 4D (x, y, z, t) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Action category label | 0D (inferred) | Fixed (inferred) |
| Multi-modal action recognition (classification) | 4D point cloud video; RGB video; action-category text descriptions | 4D (x, y, z, t); 3D (x, y, z) or (x, y, t) (inferred); 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Action category label | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers one representation-learning task and two classification tasks centered on action understanding: cross-modal alignment and action recognition. Inputs span 4D point cloud videos, RGB videos, and action-category text descriptions, so supported input dimensions are 4D (x, y, z, t), 3D video, and 1D text sequences. Outputs are fixed-size embedding/similarity objects for contrastive training and single action labels for recognition, with 0D output dimension for both. Based on the described clip/frame sampling and encoder design, dynamics are Fixed, attention is Static, and state is Constructed (inferred).

## Evidence
### Task: Cross-modal contrastive alignment (representation learning)
- "To jointly fine-tune this 4D encoder, we put forth a cross-modal contrastive learning approach, which facilitates harmonious alignment of the 3D representation with the VLM feature domain." (Section I. Introduction)
- "we introduce a cross-modal learning objective to jointly optimize the correlation alignment across language, RGB video, and 4D point cloud." (Section B. Cross-Modal Learning)
- "Our cross-modal contrastive learning mainly jointly optimizes the correlation alignment across languages, images, and point clouds via semantic-level language-4D alignment and instance-level image-4D alignment." (Section B. Cross-Modal Learning)
- Inference: In Dimension is inferred from the explicit triplet modalities (4D point cloud, RGB video, text). In Dynamics is inferred as Fixed from "we sample 2048 points in each frame," "we set the clip length and frame sampling stride to 23 and 2," and "we set the number of input frames to 8" (Section IV. Implementation Details). Attention Dynamic is inferred as Static because the model processes predefined sampled clips/text through encoders. State Dynamic is inferred as Constructed from explicit embedding construction ("embed the three modalities into text feature ... image feature ... and point cloud feature") and common embedding projection (Section B. Cross-Modal Learning; Section A. Overview of VG4D). Output/Out Dimension/Out Dynamics are inferred from fixed-size embedding/similarity outputs.

### Task: 4D point cloud action recognition (classification)
- "Experiments demonstrate that our method achieves state-ofthe-art performance for action recognition on both the NTU RGB+D 60 dataset and the NTU RGB+D 120 dataset." (Abstract)
- "**4D encoder** takes in a point cloud video as the input." (Section A. Overview of VG4D)
- "Our im-PSTNet outperforms other single modal baseline methods under most of the settings on both datasets, which demonstrates the effectiveness of our im-PSTNet for 4D action recognition on large-scale datasets." (Section IV-A. Comparison with state-of-the-art methods)
- Inference: In Dynamics is inferred as Fixed from fixed point/frame sampling settings in implementation details. Attention Dynamic is inferred as Static because the encoder consumes a predefined sampled clip. State Dynamic is inferred as Constructed from the explicit learned feature representation ("The output of the 4D encoder is a feature vector that encapsulates motion details.") (Section A. Overview of VG4D). Output/Out Dimension/Out Dynamics are inferred as fixed single-label classification.

### Task: Multi-modal action recognition (classification)
- "Building on the foundation of VG4D, we synergize the exceptional capabilities of Vision-Language Models (VLMs) in video understanding with 4D point cloud representation to enhance multi-modal action recognition." (Section I. Introduction)
- "after aligning across multiple modalities, we achieve robust multi-modal action recognition by integrating multi-modal prediction scores and utilizing text information as classifiers." (Section I. Introduction)
- "our proposed VG4D framework consists of 3 networks: 4D point cloud encoder  $E_P$ , video encoder  $E_V$  and text encoder  $E_T$  from VLM. We use language-RGB-4D point cloud triplets to train the framework." (Section A. Overview of VG4D)
- "In the testing phase, we ensemble the im-PSTNet with the VLM. Specifically, we fuse four 4D-text, RGB-text, 4D, and RGB scores as the final classification result." (Section B. Cross-Modal Learning)
- Inference: In Dimension for RGB video and text is inferred from modality type, using glossary labels 3D (x, y, z) or (x, y, t) and 1D (t). In Dynamics is inferred as Fixed from fixed clip/frame settings (Section IV. Implementation Details). Attention Dynamic is inferred as Static because inputs are predefined sampled clips/text and no runtime retrieval policy is described. State Dynamic is inferred as Constructed from explicit cross-modal embedding construction and score fusion. Output/Out Dimension/Out Dynamics are inferred as fixed single-label classification.
