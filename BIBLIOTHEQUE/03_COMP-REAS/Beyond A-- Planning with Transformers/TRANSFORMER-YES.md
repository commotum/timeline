# Beyond $A^*$ : Better Planning with Transformers via Search Dynamics Bootstrapping (Year not specified)
Source: Beyond A-- Planning with Transformers.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract states the main method is an encoder-decoder Transformer trained to predict A* search dynamics, making Transformer self-attention central to the model used for main results.
- Auxiliary analyses (TASK_MODEL_RATIO and TASK-DOMAINS) consistently describe trained encoder-decoder Transformer models across the paper’s core tasks.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files already provide sufficient direct evidence.

## Evidence
- "This is accomplished by training an encoder-decoder Transformer model to predict the search dynamics of the  $A^*$  search algorithm." (Abstract in `Beyond A-- Planning with Transformers.md`, line 7)
- "In the first experiment set, we train a set of encoder-decoder Transformer models to predict optimal plans for maze navigation tasks." (`TASK_MODEL_RATIO.md`, line 9)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for high-confidence TRANSFORMER-YES from abstract, `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Not needed because Pass 1 evidence was sufficient.
