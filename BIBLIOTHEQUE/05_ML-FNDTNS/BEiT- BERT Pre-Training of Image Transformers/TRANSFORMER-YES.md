# BEIT: BERT Pre-Training of Image Transformers (Year not specified)
Source: BEiT- BERT Pre-Training of Image Transformers.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly defines BEIT as an image-Transformer model and states masked image modeling is used to pretrain vision Transformers.
- Auxiliary analyses (TASK-DOMAINS.md and TASK_MODEL_RATIO.md) consistently identify a Transformer backbone as central to pretraining and downstream results; the extending-dimensions file was unavailable.

## Evidence
- "We introduce a self-supervised vision representation model **BEIT**, which stands for **B**idirectional **E**ncoder representation from Image Transformers." (Abstract, BEiT- BERT Pre-Training of Image Transformers.md)
- "Then we randomly mask some image patches and fed them into the backbone Transformer." (Abstract, BEiT- BERT Pre-Training of Image Transformers.md)
- "The paper's setups use fixed-resolution patch grids and a standard Transformer encoder" (Summary, TASK-DOMAINS.md)
- "After pre-training BEIT, we append a task layer upon the Transformer, and fine-tune the parameters on downstream tasks" (Quoted evidence, TASK_MODEL_RATIO.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence TRANSFORMER-YES decision from abstract + TASK-DOMAINS.md + TASK-DOMAINS.csv + TASK_MODEL_RATIO.md; Extending-dimensions analysis markdown was unavailable.
Pass 2 (targeted source scan): skipped - Pass 1 already sufficient.
