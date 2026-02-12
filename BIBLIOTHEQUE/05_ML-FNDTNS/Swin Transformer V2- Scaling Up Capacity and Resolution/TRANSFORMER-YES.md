# Swin Transformer V2: Scaling Up Capacity and Resolution (2022)
Source: Swin Transformer V2- Scaling Up Capacity and Resolution.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly identifies the central architecture as "Swin Transformer V2" and describes attention-level modifications, indicating Transformer-style self-attention is core to the method.
- Auxiliary files consistently frame the paper around the Swin Transformer model family across tasks; the extending-dimensions analysis file was unavailable (`MISSING`) but did not change the decision.

## Evidence
- "Through these techniques, this paper successfully trained a 3 billion-parameter Swin Transformer V2 model, which is the largest dense vision model to date, and makes it capable of training with images of up to 1,536×1,536 resolution." (Swin Transformer V2- Scaling Up Capacity and Resolution.md, Abstract, line 9)
- "Three main techniques are proposed: 1) a residual-post-norm method combined with cosine attention to improve training stability;" (Swin Transformer V2- Scaling Up Capacity and Resolution.md, Abstract, line 9)
- "# Swin Transformer V2: Scaling Up Capacity and Resolution (2022)" (TASK-DOMAINS.md, line 1)
- "> \"It set new performance records on 4 representative vision tasks, including ImageNet-V2 image classification, COCO object detection, ADE20K semantic segmentation, and Kinetics-400 video action classification.\" (Abstract)" (TASK_MODEL_RATIO.md, line 3)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence from abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md supports a high-confidence TRANSFORMER-YES decision.
Pass 2 (targeted source scan): skipped - Pass 1 was already sufficient; extending-dimensions analysis file was unavailable (`MISSING`).
