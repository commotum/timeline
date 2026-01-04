# Sharpness-Aware Minimization for Efficiently Improving Generalization (2021)
Source: 77aa88-2020.pdf

## Core reasons
- Introduces a new training/optimization procedure (SAM) to improve generalization by minimizing loss value and loss sharpness.
- Frames the method as a sharpness-aware, min-max optimization objective over neighborhoods in parameter space, indicating a contribution to training methodology rather than new architectures or datasets.

## Evidence extracts
- "we introduce a novel, effective procedure for in-
  stead simultaneously minimizing loss value and loss sharpness. In particular,
  our procedure, Sharpness-Aware Minimization (SAM), seeks parameters that lie
  in neighborhoods having uniformly low loss; this formulation results in a min-
  max optimization problem on which gradient descent can be performed efﬁ-
  ciently." (p. 1)
- "• We introduce Sharpness-Aware Minimization (SAM), a novel procedure that improves
  model generalization by simultaneously minimizing loss value and loss sharpness." (p. 2)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
