# Solving olympiad geometry without human demonstrations (2024)
Source: Solving olympiad geometry without human demonstrations (AlphaGeometry).md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes a central neural language model that guides proof search in the main AlphaGeometry system.
- Auxiliary analysis explicitly identifies deployed "copy of the transformer language model" in the core method; the extending-dimensions file was unavailable (`MISSING`) but not required for a confident decision.

## Evidence
- "AlphaGeometry is a neuro-symbolic system that uses a neural language model" (Abstract, `Solving olympiad geometry without human demonstrations (AlphaGeometry).md`)
- "each hosting a copy of the transformer language model" (Methods quote in `TASK_MODEL_RATIO.md`, section context: `Language model architecture and training`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence Transformer classification from abstract + `TASK-DOMAINS.md` + `TASK-DOMAINS.csv` + `TASK_MODEL_RATIO.md`; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient to finalize.
