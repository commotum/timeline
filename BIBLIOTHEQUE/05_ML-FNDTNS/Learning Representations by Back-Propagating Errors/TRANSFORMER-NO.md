# Learning representations by back-propagating errors (1986)
Source: Learning Representations by Back-Propagating Errors.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes a classical layered feed-forward backpropagation network (input, hidden, output layers) and gradient-descent weight updates, not self-attention.
- Auxiliary analyses characterize tasks as fixed/static mappings with constructed hidden states and no Transformer-style attention mechanism.

## Evidence
- "We describe a new learning procedure, back-propagation, for networks of neurone-like units." (Learning Representations by Back-Propagating Errors.md, abstract section)
- "The simplest form of the learning procedure is for layered networks which have a layer of input units at the bottom; any number of intermediate layers; and a layer of output units at the top." (Learning Representations by Back-Propagating Errors.md, abstract/body lead-in)
- "Symmetry detection (classification) ... Attention Dynamic | Static (inferred)" (TASK-DOMAINS.md, Task Table)
- "\frac{2\ \text{tasks}}{2\ \text{models}} = 1" with separate non-attention backprop task models (TASK_MODEL_RATIO.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence NON-Transformer classification; extending-dimensions analysis markdown was unavailable (MISSING).
Pass 2 (targeted source scan): skipped - Pass 1 already provided clear architecture cues (layered backprop network, no self-attention).
