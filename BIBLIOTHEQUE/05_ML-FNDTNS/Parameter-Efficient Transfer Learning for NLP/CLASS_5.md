# Parameter-Efficient Transfer Learning for NLP (2019)
Source: Parameter-Efficient Transfer Learning for NLP.md

## Core reasons
- Proposes adapter modules as a parameter-efficient transfer learning method, focusing on training only a small number of task-specific parameters while keeping the base model fixed.
- The contribution centers on a training/finetuning strategy for NLP models rather than positional encoding changes, dimensional lifting, computation mechanisms, or new datasets.

## Evidence extracts
- "we propose transfer with adapter modules. Adapter modules yield a compact and extensible model; they add only a few trainable parameters per task" (Abstract)
- "adapter tuning strategy involves injecting new layers into the original network. The weights of the original network are untouched, whilst the new adapter layers are initialized at random." (Section 2)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
