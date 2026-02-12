# Relational recurrent neural networks (2018)
Source: Relational recurrent neural networks.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract states that the central proposed model (RMC) "employs multi-head dot product attention," which is a Transformer-style self-attention mechanism and is core to the paper’s main method/results.
- Auxiliary analyses (`TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, `TASK_MODEL_RATIO.md`) are consistent with attention-centric architecture cues; the extending-dimensions analysis file was unavailable (`MISSING`) but not needed to reach high confidence.

## Evidence
- "We then improve upon these deficits by using a new memory module – a *Relational Memory Core* (RMC) – which employs multi-head dot product attention to allow memories to interact." (Abstract, `Relational recurrent neural networks.md`)
- "Using MHDPA, each memory will attend over all of the other memories, and will update its content based on the attended information." (Quoted in `TASK-DOMAINS.md`, Evidence section; original context: Section 3.1)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for high-confidence TRANSFORMER-YES from abstract plus all available auxiliary files; extending-dimensions file marked unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was already sufficient for final decision.
