# Lost in the Middle: How Language Models Use Long Contexts (Year not specified)
Source: Lost in the Middle- How Language Models Use Long Contexts.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The source explicitly states the studied language models are Transformer-based.
- Core experiments analyze decoder-only/encoder-decoder behavior via token-level attention access, making attention central to the main results.
- The auxiliary model-ratio analysis indicates the same model set is reused across tasks, so this architecture is central rather than a peripheral baseline.
- The Extending-dimensions analysis file was unavailable (`MISSING`), but available evidence is sufficient for a high-confidence decision.

## Evidence
- "Existing language models are generally implemented with Transformers (Vaswani et al., 2017)." (Lost in the Middle- How Language Models Use Long Contexts.md, §1 Introduction, line 21)
- "The open models we evaluated are all decoder-only models—at each timestep, they may only attend to prior tokens." (Lost in the Middle- How Language Models Use Long Contexts.md, §4.1 Effect of Model Architecture, line 206)
- "We use the same set of models as the multi-document question answering experiments, see §2.2 for more details." (TASK_MODEL_RATIO.md, item 2)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - task framing and model-family cues were strong, but explicit architecture evidence for self-attention centrality was limited.
Pass 2 (targeted source scan): performed - found direct Transformer and attention-based architecture statements sufficient to finalize.
