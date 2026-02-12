# Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Year not specified)
Source: Direct Preference Optimization- Your Language Model is Secretly a Reward Model.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract frames the method around fine-tuning large language models, and the auxiliary model file identifies GPT-2/GPT-J/Pythia model families, which are Transformer-based LMs.
- The main experimental results are produced by training these LMs with DPO, so Transformer self-attention is materially central rather than peripheral.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the available abstract and auxiliary evidence is sufficient for a high-confidence decision.

## Evidence
- "Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods." (Abstract, `Direct Preference Optimization- Your Language Model is Secretly a Reward Model.md`)
- "For SFT, we fine-tune GPT-2-large until convergence on reviews from the train split of the IMDB dataset (further details in App C.1)." (`TASK_MODEL_RATIO.md`, item 2)
- "DPO, PPO and Preferred-FT all fine-tune the same GPT-J SFT model<sup>4</sup>." (`TASK_MODEL_RATIO.md`, item 2)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence decision; extending-dimensions input was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided sufficient evidence.
