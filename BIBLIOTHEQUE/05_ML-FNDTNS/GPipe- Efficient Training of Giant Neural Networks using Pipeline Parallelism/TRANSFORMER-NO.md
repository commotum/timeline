# GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism (Year not specified)
Source: GPipe- Efficient Training of Giant Neural Networks using Pipeline Parallelism.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The paper's core contribution is GPipe as architecture-agnostic pipeline parallelism for "any network that can be expressed as a sequence of layers," not a Transformer architecture.
- Transformer is one of two evaluation models (the other is AmoebaNet/CNN), so self-attention is not the central model basis of the paper's main method.

## Evidence
- "we introduce GPipe, a pipeline parallelism library that allows scaling any network that can be expressed as a sequence of layers." (Abstract, file: GPipe- Efficient Training of Giant Neural Networks using Pipeline Parallelism.md)
- "We demonstrate the advantages of GPipe by training large-scale neural networks on two different tasks with distinct network architectures: (i) Image Classification: We train a 557-million-parameter AmoebaNet model and attain a top-1 accuracy of 84.4% on ImageNet-2012, (ii) Multilingual Neural Machine Translation: We train a single 6-billion-parameter, 128-layer Transformer model on a corpus spanning over 100 languages and achieve better quality than all bilingual models." (Abstract, file: GPipe- Efficient Training of Giant Neural Networks using Pipeline Parallelism.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for high-confidence NO; TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md were read in full; Extending-dimensions analysis file was unavailable (MISSING).
Pass 2 (targeted source scan): skipped - Pass 1 already provided clear architecture-centrality evidence.
