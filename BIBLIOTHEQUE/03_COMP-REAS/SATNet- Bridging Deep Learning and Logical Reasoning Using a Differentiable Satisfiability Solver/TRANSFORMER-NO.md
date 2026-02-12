# SATNet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver (2019)
Source: SATNet- Bridging Deep Learning and Logical Reasoning Using a Differentiable Satisfiability Solver.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract and auxiliary analyses consistently describe SATNet as a differentiable MAXSAT/SDP solver layer, not a Transformer or self-attention architecture.
- The only deep vision component described is a "traditional convolutional architecture"/CNN feeding SATNet for visual Sudoku; no Transformer block is central.

## Evidence
- "we propose a new direction toward this goal by introducing a differentiable (smoothed) maximum satisfiability (MAXSAT) solver that can be integrated into the loop of larger deep learning systems." (SATNet- Bridging Deep Learning and Logical Reasoning Using a Differentiable Satisfiability Solver.md, Abstract)
- "We also solve a \"visual Sudoku\" problem ... by combining our MAXSAT solver with a traditional convolutional architecture." (SATNet- Bridging Deep Learning and Logical Reasoning Using a Differentiable Satisfiability Solver.md, Abstract)
- "attention is static, and state is constructed through SATNet's continuous relaxations, auxiliary variables, and iterative coordinate-descent inference." (TASK-DOMAINS.md, Summary)
- "Our architecture for this problem uses a convolutional neural network connected to a SATNet layer." (TASK_MODEL_RATIO.md, Section 4.3 quote)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for high-confidence TRANSFORMER-NO from abstract + TASK-DOMAINS.md + TASK-DOMAINS.csv + TASK_MODEL_RATIO.md; Extending-dimensions file was unavailable (MISSING).
Pass 2 (targeted source scan): skipped - Pass 1 already decisive.
