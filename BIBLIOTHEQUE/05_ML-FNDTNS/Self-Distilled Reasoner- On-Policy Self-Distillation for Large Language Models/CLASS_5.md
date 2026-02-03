# Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models (Not specified in the paper)
Source: Self-Distilled Reasoner- On-Policy Self-Distillation for Large Language Models.md

## Core reasons
- Proposes an on-policy self-distillation training framework where the same LLM serves as teacher and student, which is a training/optimization contribution rather than positional encoding or dimensional adaptation.
- Focuses on improving reasoning via distillation objectives and dense supervision on student rollouts, not on creating a new dataset or benchmark as the main contribution.

## Evidence extracts
- "we introduce *On-Policy* Self-Distillation (OPSD), a framework where a single model acts as both teacher and student by conditioning on different contexts." (Abstract)
- "training minimizes the pertoken divergence between these distributions over the student's own rollouts." (Abstract)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
