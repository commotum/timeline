# ARC-NCA: Towards Developmental Solutions to the Abstraction and Reasoning Corpus (Year not specified)
Source: ARC-NCA- Towards Developmental Solutions to the Abstraction and Reasoning Corpus.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: medium
Basis: abstract-aux-only

## Why
- The abstract frames the core method as Neural Cellular Automata (NCA) and EngramNCA, not a Transformer-family architecture.
- The available auxiliary analyses consistently describe CA/CNN-style local update models and per-task CA training; the extending-dimensions file was unavailable (`MISSING`).

## Evidence
- "This paper introduces ARC-NCA, a developmental approach leveraging standard Neural Cellular Automata (NCA) and NCA enhanced with hidden memories (EngramNCA) to tackle the ARC-AGI benchmark." (Abstract, `ARC-NCA- Towards Developmental Solutions to the Abstraction and Reasoning Corpus.md`)
- "We take this to mean that our program generator, the system that trains NCAs, can train a new CA per problem." (Section: **Model Training**, quoted in `TASK_MODEL_RATIO.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Read abstract and all available auxiliary files (`TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, `TASK_MODEL_RATIO.md`); extending-dimensions analysis was unavailable (`MISSING`); evidence was sufficient to decide.
Pass 2 (targeted source scan): skipped - Pass 1 provided sufficient evidence for the binary decision.
