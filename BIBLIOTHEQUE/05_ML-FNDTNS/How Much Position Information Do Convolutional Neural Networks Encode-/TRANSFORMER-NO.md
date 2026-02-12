# How Much Position Information Do Convolutional Neural Networks Encode? (Year not specified)
Source: How Much Position Information Do Convolutional Neural Networks Encode-.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The paper’s core method is a CNN-based position encoding analysis pipeline (PosENet with VGG/ResNet encoders), not a Transformer or self-attention architecture.
- Auxiliary analyses also point to convolutional model families only, and the extending-dimensions file was unavailable (`MISSING`), with no contrary evidence in available sources.

## Evidence
- "In contrast to fully connected networks, Convolutional Neural Networks (CNNs) achieve efficiency by learning weights associated with local filters with a finite spatial extent." (Abstract, `How Much Position Information Do Convolutional Neural Networks Encode-.md`:9)
- "Our Position Encoding Network (PosENet) (See Fig. 2) consists of two key components: a feed-forward convolutional encoder network  $f_{enc}$  and a simple position encoding module, denoted as  $f_{pem}$ ." (Section 2.1, `How Much Position Information Do Convolutional Neural Networks Encode-.md`:39)
- "we train the VGG and ResNet based networks on each type of the ground-truth" (`TASK_MODEL_RATIO.md`:9)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for TRANSFORMER-NO from abstract + `TASK-DOMAINS.md` + `TASK-DOMAINS.csv` + `TASK_MODEL_RATIO.md`; extending-dimensions analysis markdown was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided high-confidence architecture evidence.
