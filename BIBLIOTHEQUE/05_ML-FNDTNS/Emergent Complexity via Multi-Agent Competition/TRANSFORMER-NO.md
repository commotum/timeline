# Emergent Complexity via Multi-Agent Competition (Year not specified)
Source: Emergent Complexity via Multi-Agent Competition.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The available auxiliary analyses indicate PPO policies implemented with MLP/LSTM architectures, not Transformer/self-attention blocks.
- Architecture cues in the auxiliary evidence explicitly name MLP and single-layer LSTM policies as central for the main tasks.
- The Extending-dimensions analysis markdown was unavailable (`MISSING`), so the decision uses the abstract plus available auxiliary files.

## Evidence
- "MLP policy and value functions for the run-to-goal and you-shall-not-pass" and "LSTM policy and value function for sumo and kick-and-defend" (`TASK-DOMAINS.md`, line 32; inference quote citing Section 5.1)
- "single-layer LSTM with 128 hidden state dimension" (`TASK-DOMAINS.md`, line 46; inference quote citing Section 5.1)
- "Reinforcement learning algorithms can train agents that solve problems in complex, interesting environments." (`Emergent Complexity via Multi-Agent Competition.md`, Abstract, line 7)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - abstract reviewed; `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md` read in full; evidence sufficient for high-confidence NO.
Pass 2 (targeted source scan): skipped - not needed after Pass 1.
