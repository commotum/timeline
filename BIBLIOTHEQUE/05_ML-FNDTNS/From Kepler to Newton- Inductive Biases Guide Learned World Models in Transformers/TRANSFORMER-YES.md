# From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers (Year not specified)
Source: From Kepler to Newton- Inductive Biases Guide Learned World Models in Transformers.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract makes Transformers the central model family and reports successful world-model learning after adding inductive biases.
- A core reported mechanism is changing Transformer attention behavior (temporal locality via restricted attention window), which is architectural and central to the main results.
- Auxiliary analyses consistently identify Transformer/GPT-style training on the main tasks; the extending-dimensions file was unavailable (`MISSING`) but the remaining evidence was sufficient.

## Evidence
- "We show that ensuring **spatial smoothness** (by formulating prediction as continuous regression) and stability (by training with noisy contexts to mitigate error accumulation) enables generic Transformers to surpass prior failures and learn a coherent **Keplerian** world model, successfully fitting ellipses to planetary trajectories." (Abstract, From Kepler to Newton- Inductive Biases Guide Learned World Models in Transformers.md)
- "By restricting the attention window to the immediate past-imposing the simple assumption that future states depend only on the local state rather than a complex history—we force the model to abandon curve-fitting and discover Newtonian force representations." (Abstract, From Kepler to Newton- Inductive Biases Guide Learned World Models in Transformers.md)
- "Vafa et al. (2025) trained a GPT-2-scale transformer model to predict planetary motion." (TASK-DOMAINS.md, Evidence: Planetary motion trajectory prediction, citing Section 2.1)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence TRANSFORMER-YES decision; extending-dimensions analysis markdown was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - abstract plus auxiliary files were already decisive.
