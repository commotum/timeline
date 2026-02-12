# Dota 2 with Large Scale Deep Reinforcement Learning (2021)
Source: Dota 2 with Large Scale Deep Reinforcement Learning.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The auxiliary analysis identifies the central policy/value architecture as LSTM-based, not Transformer-based.
- The abstract and task/model-ratio evidence describe large-scale self-play RL training and a single shared policy, with no material self-attention component; `EXTENDING-DIMENSIONS.md` was unavailable.

## Evidence
- "The neural network consists primarily of a single-layer 4096-unit LSTM" (TASK-DOMAINS.md, Evidence section quoting Section 3.1 Playing Dota using AI)
- "In addition to the action logits, the value function is computed as another linear projection of the LSTM state." (TASK-DOMAINS.md, Evidence section quoting Section H Neural Network Architecture)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for high-confidence classification; abstract reviewed, TASK-DOMAINS.md/TASK-DOMAINS.csv/TASK_MODEL_RATIO.md read in full, and Extending-dimensions analysis file was unavailable.
Pass 2 (targeted source scan): skipped - Pass 1 evidence was already sufficient and unambiguous.
