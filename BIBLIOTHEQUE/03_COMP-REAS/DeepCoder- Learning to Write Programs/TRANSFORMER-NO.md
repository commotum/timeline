# DeepCoder: Learning to Write Programs (Year not specified)
Source: DeepCoder- Learning to Write Programs.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract frames the method as a neural predictor of program properties used to guide classic program-synthesis search (enumerative + SMT), not a self-attention architecture.
- The auxiliary analysis explicitly states the model uses a "simple feed-forward architecture" and reports comparison against an RNN baseline, with no Transformer-style blocks.
- The extending-dimensions file was unavailable (`MISSING`), but the abstract plus available auxiliary files already provide decisive architecture cues.

## Evidence
- "The approach is to train a neural network to predict properties of the program that generated the outputs from the inputs. We use the neural network's predictions to augment search techniques from the programming languages community, including enumerative search and an SMT-based solver." (DeepCoder- Learning to Write Programs.md, ABSTRACT)
- "Empirically, we show that our approach leads to an order of magnitude speedup over the strong non-augmented baselines and a Recurrent Neural Network approach" (DeepCoder- Learning to Write Programs.md, ABSTRACT)
- "For the encoder we use a simple feed-forward architecture." (TASK-DOMAINS.md, Evidence: Task: Multi-label classification)
- "Extending-dimensions analysis markdown: MISSING" (User-provided input specification; file unavailable)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for a high-confidence TRANSFORMER-NO decision from abstract and available auxiliary files.
Pass 2 (targeted source scan): skipped - Pass 1 was already decisive; no additional body scan needed.
