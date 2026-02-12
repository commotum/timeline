# Multi-task Sequence to Sequence Learning (Year not specified)
Source: Multi-task Sequence to Sequence Learning.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The paper focuses on multi-task sequence-to-sequence learning and does not present Transformer-style self-attention blocks as the core model.
- Auxiliary analysis identifies the system as attention-free and explicitly states attention is not employed; the extending-dimensions analysis file was unavailable (`MISSING`).

## Evidence
- "This paper examines three multi-task learning (MTL) settings for sequence to sequence models" (Multi-task Sequence to Sequence Learning.md, Abstract)
- "The models are described as attention-free encoder-decoder systems" (TASK-DOMAINS.md, Summary)
- "our sequence to sequence models do not employ the attention mechanism" (TASK-DOMAINS.md, Evidence section quoting Conclusion)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence TRANSFORMER-NO decision from abstract + TASK-DOMAINS.md + TASK-DOMAINS.csv + TASK_MODEL_RATIO.md; Extending-dimensions analysis was unavailable (MISSING).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient to finalize.
