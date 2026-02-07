# Learning to Predict by the Methods of Temporal Differences (1988)
Source: Learning to Predict by the Methods of Temporal Differences.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prediction (multi-step outcome) | observation vector sequence (x_t) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | prediction sequence of scalar outcome z (P_t) | 1D (t) (inferred) | Capped (inferred) |
| Prediction (cumulative outcome) | observation vector sequence (x_t) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | prediction sequence of cumulative cost z_t | 1D (t) (inferred) | Capped (inferred) |
| Prediction (fixed-interval future event) | observation vector sequence (x_t) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | fixed-horizon prediction sequence (P_t^delta) | 1D (t) (inferred) | Open (inferred) |
| Prediction (discounted cumulative return) | observation vector sequence (x_t) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | prediction of discounted sum z_t | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper centers on temporal-difference methods for multi-step prediction over time-indexed sequences, producing per-time-step scalar predictions of sequence outcomes. It extends this to predicting cumulative outcomes, fixed-horizon future events, and discounted infinite-horizon returns. The described tasks operate on temporal sequences and assume predictions are computed from the observations directly, implying static attention and direct state use, with mostly finite (capped) sequences and some open-ended horizons.

## Evidence
### Task: Prediction (multi-step outcome)
- "observation-outcome sequences of the form  $x_1, x_2, x_3, \ldots, x_m, z$" (Section 2.2)
- "each  $x_t$  is a vector of observations available at time t in the sequence" (Section 2.2)
- "produces a corresponding sequence of predictions  $P_1, P_2, P_3, \ldots, P_m$" (Section 2.2)
- "z is assumed to be a real-valued scalar." (Section 2.2)
- Inference: Marked 1D (t) and capped dynamics because sequences are given as "$x_1, x_2, x_3, \ldots, x_m, z$"; static/direct because "here we assume that it is a function only of  $x_t$ ." (Section 2.2)

### Task: Prediction (cumulative outcome)
- "predict a quantity that accumulates over a sequence." (Section 5.1)
- "use the observation vector received at each step to predict the total cumulative cost after that step" (Section 5.1)
- "We would like  $P_t$  to equal the expected value of  $z_t = \sum_{k=t}^m c_{k+1}$" (Section 5.1)
- Inference: Classified inputs/outputs as 1D (t) sequences with capped dynamics and static/direct processing because "m is the number of observation vectors in the sequence." (Section 5.1)

### Task: Prediction (fixed-interval future event)
- "consider the problem of making a prediction for a particular fixed amount of time later." (Section 5.3)
- "on each Monday, you predict whether it will rain on the following Monday" (Section 5.3)
- "At each day t, we must form not only  $P_t^7$" (Section 5.3)
- "but also  $P_t^6$ ,  $P_t^5$ , ...,  $P_t^1$" (Section 5.3)
- Inference: Treated inputs/outputs as 1D (t) sequences with open dynamics and static/direct processing because predictions repeat "on each Monday" and continue "for each day of the week." (Section 5.3)

### Task: Prediction (discounted cumulative return)
- "some process generates costs  $c_{t+1}$  at each transition from t to t+1" (Section 6.4)
- "we may want  $P_t$  to predict the discounted sum:" (Section 6.4)
- "$$z_t = \sum_{k=0}^{\infty} \gamma^k c_{t+k+1},$$" (Section 6.4)
- "where  $P_t$  is the linear form  $w^T x_t$" (Section 6.4)
- Inference: Labeled inputs/outputs as 1D (t) with open dynamics and static/direct processing because the target is an infinite-horizon discounted sum (\sum_{k=0}^{\infty}). (Section 6.4)
