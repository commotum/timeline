# True Online Emphatic TD(λ): Quick Reference and Implementation Guide (2015)
Source: True Online Emphatic TD($λ$)- Quick Reference and Implementation Guide.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long-term prediction (general value function / value prediction) | time-series feature vectors ($\phi_t$) and cumulant signals ($R_t$), with policy/discount/interest/bootstrapping/step-size sequences ($\rho_t, \gamma_t, I_t, \lambda_t, \alpha_t$) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | scalar prediction $\phi_t^{\top}\theta_t$ approximating discounted cumulative outcome at each time step | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper covers a single reinforcement-learning prediction task: learning general value-function estimates (long-term predictions) from a temporal stream. Inputs and outputs are both organized over time, so the task is classified as 1D (t) under the glossary scheme. The interface is stream-oriented rather than fixed-length batch-oriented, supporting an Open dynamics classification for both input and output. The algorithm uses fixed per-step inputs (Static attention) while maintaining learned internal traces and weights (Constructed state).

## Evidence
### Task: long-term prediction (general value function / value prediction)
- "This document is a guide to the implementation of true online emphatic  $TD(\lambda)$ , a model-free temporal-difference algorithm for learning to make long-term predictions which combines the emphasis idea (Sutton, Mahmood & White 2015) and the true-online idea (van Seijen & Sutton 2014)." (Section 1 Setting and requirements)
- "The algorithm is meant to be called at regular intervals with data from a time series, from which it learns to make a prediction. The time series includes a feature vector  $\phi_t \in \mathbb{R}^n$  and a cumulant signal  $R_t \in \mathbb{R}$ ." (Section 1 Setting and requirements)
- "The prediction at each time is linear in the feature vector." (Section 1 Setting and requirements)
- "Internal to the learning algorithm are the learned weight vector,  $\boldsymbol{\theta}_t \in \mathbb{R}^n$ , and an auxiliary shorter-term-memory vector  $\boldsymbol{e}_t \in \mathbb{R}^n$  with  $\boldsymbol{e}_t \geq \boldsymbol{0}$ ." (Section 2 Algorithm Specification)
- Inference: `In Dimension` and `Out Dimension` are `1D (t)` because the paper explicitly defines the data as a "time series" and states prediction occurs "at each time." `In Dynamics` and `Out Dynamics` are `Open` because the algorithm is repeatedly called over steps and pseudocode states "On each step, t = 0, 1, 2, ..., the **learn** function is called with arguments  $\alpha_t, I_t, \lambda_t, \phi_t, \rho_t, R_{t+1}, \phi_{t+1}, \gamma_{t+1}$ :" (Section 3 Pseudocode). `Attention Dynamic` is `Static` because each step consumes that fixed argument list with no runtime selection mechanism (Section 3 Pseudocode). `State Dynamic` is `Constructed` because the algorithm maintains internal learned and memory variables (`\theta_t, e_t, M_t, F_t`) beyond raw inputs (Section 2 Algorithm Specification).
