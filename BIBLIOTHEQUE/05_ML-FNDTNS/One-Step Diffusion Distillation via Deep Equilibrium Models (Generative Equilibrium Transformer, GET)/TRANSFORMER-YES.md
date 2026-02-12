# One-Step Diffusion Distillation via Deep Equilibrium Models (Generative Equilibrium Transformer, GET) (Year not specified)
Source: One-Step Diffusion Distillation via Deep Equilibrium Models (Generative Equilibrium Transformer, GET).md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The paper’s core distilled architecture is explicitly a Transformer variant (GET), and this is positioned as central to the main one-step generation results.
- Auxiliary analyses consistently mark transformer attention as part of the core model behavior, with no indication that Transformer components are only peripheral baselines.

## Evidence
- "Of particular importance to our approach is to leverage a new Deep Equilibrium (DEQ) model as the distilled architecture: the Generative Equilibrium Transformer (GET)." (Abstract, One-Step Diffusion Distillation via Deep Equilibrium Models (Generative Equilibrium Transformer, GET).md)
- "Attention, state, and fixed-size dynamics are inferred from the transformer-based DEQ architecture that solves for a fixed-point latent representation." (Summary, TASK-DOMAINS.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for TRANSFORMER-YES from abstract + TASK-DOMAINS.md/TASK-DOMAINS.csv/TASK_MODEL_RATIO.md; Extending-dimensions analysis file was unavailable (MISSING).
Pass 2 (targeted source scan): skipped - Pass 1 already provided high-confidence architecture evidence.
