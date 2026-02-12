# Learning to summarize from human feedback (Year not specified)
Source: Learning to summarize from human feedback.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- `TASK-DOMAINS.md` explicitly indicates that the paper’s trained models are Transformer decoders in GPT-3 style, so self-attention is central to the model family used for main results.
- The abstract describes the core training pipeline (reward model + summarization policy fine-tuning) as the main method, and the auxiliary analyses map that pipeline to Transformer-based models.
- The extending-dimensions analysis file was unavailable (`MISSING`), so the decision is based on the abstract and available auxiliary files.

## Evidence
- "We collect a large, high-quality dataset of human comparisons between summaries, train a model to predict the human-preferred summary, and use that model as a reward function to fine-tune a summarization policy using reinforcement learning." (Learning to summarize from human feedback.md, Abstract)
- "All of our models are Transformer decoders [62] in the style of GPT-3 [47, 4]." (TASK-DOMAINS.md, Evidence section quoting Section 3.4 Models)
- "Attention and state dynamics are inferred as static/direct because only fixed-size Transformer decoders without external memory or retrieval are described." (TASK-DOMAINS.md, Summary)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for a high-confidence decision; abstract and auxiliary files provide explicit Transformer-decoder evidence, and `TASK-DOMAINS.csv`/`TASK_MODEL_RATIO.md` are consistent with this framing.
Pass 2 (targeted source scan): skipped - Pass 1 already provided explicit architecture evidence.
