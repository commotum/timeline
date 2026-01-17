# Improving neural networks by preventing co-adaptation of feature detectors (Not specified in the paper.)
Source: Improving neural networks by preventing co-adaptation of feature detectors.md

## Core reasons
- Proposes dropout as a training regularization method that randomly omits units to reduce overfitting and co-adaptation.
- Emphasizes generalization improvements and performance gains from the training method across benchmarks rather than new data or model families.

## Evidence extracts
- "This \"overfitting\" is greatly reduced by randomly omitting half of the feature detectors on each training case. This prevents complex co-adaptations in which a feature detector is only helpful in the context of several other specific feature detectors." (Section: Main text)
- "Overfitting can be reduced by using \"dropout\" to prevent complex co-adaptations on the training data. On each presentation of each training case, each hidden unit is randomly omitted from the network with a probability of 0.5, so a hidden unit cannot rely on other hidden units being present." (Section: Main text)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
