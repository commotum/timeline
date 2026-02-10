# You Only Look Once: Unified, Real-Time Object Detection (Not specified in the paper)
Source: You Only Look Once- Unified, Real-Time Object Detection.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| object detection | images | 2D (x, y) (inferred) | Fixed | Static (inferred) | Direct (inferred) | bounding boxes and class probabilities (detections) | 2D (x, y); 0D (inferred) | Capped (inferred) |
| classification (pretraining) | images | 2D (x, y) (inferred) | Fixed | Static (inferred) | Direct (inferred) | class probabilities over 1000 classes | 0D | Fixed |

## Summary
This paper primarily covers object detection from single images, with an explicit pretraining stage on image classification. The supported input modality is 2D image grids with fixed-size model interfaces (448x448 for detection and 224x224 during classification pretraining). Output structure spans capped multi-object detections for detection and fixed-size class outputs for pretraining classification. Attention behavior is static and state behavior is direct by inference from the single-pass feedforward design.

## Evidence
### Task: object detection
- "we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities." (Section Abstract)
- "A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation." (Section Abstract)
- "Our system (1) resizes the input image to  $448 \times 448$ , (2) runs a single convolutional network on the image, and (3) thresholds the resulting detections by the model's confidence." (Section 1, Figure 1 caption)
- "On PASCAL VOC the network predicts 98 bounding boxes per image and class probabilities for each box." (Section 2.3. Inference)
- Inference: In Dimension is labeled as `2D (x, y)` because the task operates on images and predicts spatial box coordinates; Attention Dynamic is `Static (inferred)` and State Dynamic is `Direct (inferred)` because the paper describes a single fixed network evaluation over the full resized image; Out Dynamics is `Capped (inferred)` because detections are thresholded and constrained by a fixed maximum of 98 box predictions (Section 1 Figure 1 caption; Section 2.3. Inference).

### Task: classification (pretraining)
- "Our network architecture is inspired by the GoogLeNet model for image classification [34]." (Section 2.1. Network Design)
- "We pretrain our convolutional layers on the ImageNet 1000-class competition dataset [30]." (Section 2.2. Training)
- "For pretraining we use the first 20 convolutional layers from Figure 3 followed by a average-pooling layer and a fully connected layer." (Section 2.2. Training)
- "We train this network for approximately a week and achieve a single crop top-5 accuracy of 88% on the ImageNet 2012 validation set" (Section 2.2. Training)
- Inference: In Dimension is `2D (x, y) (inferred)` from image-based pretraining, and Attention/State are `Static (inferred)` / `Direct (inferred)` from the same fixed feedforward setup described for YOLO's network design and training pipeline (Section 2.1; Section 2.2).
