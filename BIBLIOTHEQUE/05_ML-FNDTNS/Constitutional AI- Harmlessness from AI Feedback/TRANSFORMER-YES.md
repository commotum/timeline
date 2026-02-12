# Constitutional AI: Harmlessness from AI Feedback (2022)
Source: Constitutional AI- Harmlessness from AI Feedback.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: medium
Basis: source-targeted-scan

## Why
- The paper materials describe the core system as a general-purpose/large language model assistant (including Claude) used for the main CAI/RLAIF results; this model family is Transformer-based in practice.
- The extending-dimensions analysis file was unavailable (`MISSING`), so the decision relies on the abstract plus available auxiliary files and targeted architecture-cue scanning, with no non-Transformer architecture cues found.

## Evidence
- "We want general-purpose language models to be as useful as possible and we want them to be safe." (Constitutional AI- Harmlessness from AI Feedback.md:7)
- "1 Note that this was for research purposes, and is not the same set of principles that Anthropic uses for its large language model, Claude." (Constitutional AI- Harmlessness from AI Feedback.md:47)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Abstract + `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md` read in full; they indicate an LLM assistant setting, while the extending-dimensions file is unavailable (`MISSING`).
Pass 2 (targeted source scan): performed - Scanned the source for architecture cues; found LLM/Claude cues and no evidence of a non-Transformer central model.
