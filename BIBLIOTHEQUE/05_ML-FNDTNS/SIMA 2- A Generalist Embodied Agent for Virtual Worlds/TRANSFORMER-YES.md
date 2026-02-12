# SIMA 2: A Generalist Embodied Agent for Virtual Worlds (2025)
Source: SIMA 2- A Generalist Embodied Agent for Virtual Worlds.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract states the agent is "Built upon a Gemini foundation model," and Gemini-family models are Transformer-based architectures where self-attention is central.
- Auxiliary analyses consistently describe SIMA 2 as a Gemini-core model (including Gemini Flash-Lite), making Transformer-style self-attention part of the central model rather than a peripheral baseline.
- The Extending-dimensions analysis markdown was unavailable (`MISSING`), but the abstract plus available auxiliary files already provide sufficient architecture evidence.

## Evidence
- "Built upon a Gemini foundation model, SIMA 2 represents a significant step toward active, goal-directed interaction within an embodied environment." (SIMA 2- A Generalist Embodied Agent for Virtual Worlds.md, abstract, line 7)
- "At its core, the SIMA 2 agent architecture is a Gemini Flash-Lite model" (TASK_MODEL_RATIO.md, line 9; cites Section 3.3)
- "Embodied Dialogue SIMA 2 is, at its core, a Gemini model." (TASK-DOMAINS.md, line 26; cites Section 4.1)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence Transformer-core classification from abstract + TASK-DOMAINS.md + TASK-DOMAINS.csv + TASK_MODEL_RATIO.md; Extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was already sufficient for a high-confidence decision.
