# AI safety via debate (Year not specified)
Source: AI safety via debate.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: hint-only

## Why
- The hint files describe the evaluated ML setup as a sparse MNIST classifier plus debate mechanics, with no Transformer or self-attention architecture indicated.
- The hints explicitly describe debate play via Monte Carlo Tree Search in the shown experiment, which is not a Transformer-style self-attention model.

## Evidence
- "Concretely, the judge is trained to classify MNIST from 6 (resp. 4) nonzero pixels, with the pixels chosen at random at training time." (TASK_MODEL_RATIO.md, quote from Section 3.1)
- "The MNIST debate game is simple enough that we can play it with pure Monte Carlo Tree Search [Coulom, 2006] without training a heuristic as in Silver et al. [2017a]." (TASK_MODEL_RATIO.md, quote from Section 3.1)

## Pass accounting
Pass 0 (hint-first): performed - sufficient evidence for a high-confidence non-Transformer decision from TASK_MODEL_RATIO.md/TASK-DOMAINS.*
Pass 1 (source triage): skipped - hint evidence was sufficient
Pass 2 (source deep dive): skipped - not needed after Pass 0
