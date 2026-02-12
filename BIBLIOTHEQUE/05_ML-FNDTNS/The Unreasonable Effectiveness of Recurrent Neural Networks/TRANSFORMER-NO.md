# The Unreasonable Effectiveness of Recurrent Neural Networks (2015)
Source: The Unreasonable Effectiveness of Recurrent Neural Networks.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The paper's opening abstract-style text and method description are explicitly centered on recurrent neural networks and LSTMs, not Transformer blocks.
- Auxiliary analyses consistently mark attention as static and recurrent state updates as direct (`attention_dynamic=Static`, `state_dynamic=Direct`), with no Transformer/self-attention architecture indicated as central.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the available abstract + auxiliary evidence is already sufficient for a high-confidence decision.

## Evidence
- "There's something magical about Recurrent Neural Networks (RNNs)." (The Unreasonable Effectiveness of Recurrent Neural Networks.md:5, opening abstract-style paragraph)
- "language models based on multi-layer LSTMs." (The Unreasonable Effectiveness of Recurrent Neural Networks.md:19, opening abstract-style paragraph)
- "Attention and state are inferred as Static and Direct for these examples based on the paper’s fixed recurrent update framing and next-step prediction setup." (TASK-DOMAINS.md:14)
- "Image captioning,images,2D (x, y) (inferred),Fixed (inferred),Static (inferred),Direct (inferred)" (TASK-DOMAINS.csv:2)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for high-confidence TRANSFORMER-NO using the opening abstract-style section plus TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient for final classification.
