# You Only Look Once: Unified, Real-Time Object Detection (Year not specified)
Source: You Only Look Once- Unified, Real-Time Object Detection.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes YOLO as a single feedforward detection network and does not indicate Transformer-style self-attention as a core mechanism.
- The auxiliary files consistently describe a convolutional, static-attention setup; the extending-dimensions analysis file was unavailable (`MISSING`).

## Evidence
- "A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation." (You Only Look Once- Unified, Real-Time Object Detection.md, Abstract)
- "Our system (1) resizes the input image to  $448 \times 448$ , (2) runs a single convolutional network on the image, and (3) thresholds the resulting detections by the model's confidence." (TASK-DOMAINS.md, Evidence, Task: object detection)
- "\"object detection\",\"images\",\"2D (x, y) (inferred)\",\"Fixed\",\"Static (inferred)\",\"Direct (inferred)\",\"bounding boxes and class probabilities (detections)\",\"2D (x, y); 0D (inferred)\",\"Capped (inferred)\"" (TASK-DOMAINS.csv, row: object detection)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for high-confidence TRANSFORMER-NO from abstract and auxiliary files; extending-dimensions file was unavailable (MISSING).
Pass 2 (targeted source scan): skipped - Pass 1 was already sufficient for a high-confidence decision.
