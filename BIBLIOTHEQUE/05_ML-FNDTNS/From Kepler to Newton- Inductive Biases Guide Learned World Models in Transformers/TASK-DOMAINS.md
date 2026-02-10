# From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers (Year not specified in the paper.)
Source: From Kepler to Newton- Inductive Biases Guide Learned World Models in Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1D harmonic trajectory next-token prediction | tokenized 1D sine-wave trajectories | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | next coordinate token in the trajectory | 1D (t) (inferred) | Fixed (inferred) |
| Planetary motion trajectory prediction (next-state / next-token) | planetary position trajectories (continuous (x, y) coordinates or discretized coordinate tokens) | 3D (x, y, t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | predicted next/future planetary positions | 3D (x, y, t) (inferred) | Capped (inferred) |

## Summary
The paper covers autoregressive prediction tasks over physical trajectories in both a simplified 1D harmonic setting and a 2D planetary-orbit setting. The 1D sine-wave setup is framed as next-token prediction, while planetary motion is handled as both next-token prediction and next-state regression. From the OCR evidence, the task domains span 1D (t) and 3D (x, y, t) representations (inferred), with fixed or capped interface sizes depending on sequence/context setup. Attention is static and state is direct (inferred) because the model operates on predefined contexts in reactive autoregressive prediction.

## Evidence
### Task: 1D harmonic trajectory next-token prediction
- "To simplify the setting while retaining the essential features of tokenization, we adopt a 1D sine-wave dataset, which qualitatively resembles the oscillatory behavior of planetary motion but reduces the problem to one dimension." (Section 2.3. Conditions and scaling laws for spatial map emergence)
- "We choose  $\Delta t = 0.2$ and T=20, yielding  $T/\Delta t=100$  points per trajectory." (Section 2.3. Conditions and scaling laws for spatial map emergence)
- "Since  $x \in [-1, 1]$ , we partition this range uniformly into V bins/tokens, converting each trajectory into a sequence of token IDs, e.g.,  $[6, 12, 17, 20, 21, 19, \ldots]$ . Transformer models are trained using next-token prediction with cross-entropy loss." (Section 2.3. Conditions and scaling laws for spatial map emergence)
- Inference: In/Out Dimension are inferred as 1D (t) from the one-dimensional trajectory description and token sequence formulation; In/Out Dynamics are inferred as Fixed from "100 points per trajectory." Attention Dynamic is inferred as Static and State Dynamic as Direct from the fixed autoregressive next-token setup without runtime retrieval or explicit external state construction.

### Task: Planetary motion trajectory prediction (next-state / next-token)
- "Vafa et al. (2025) trained a GPT-2-scale transformer model to predict planetary motion." (Section 2.1. Problem setup)
- "$$(x_{i+1}, y_{i+1}) = f_{\theta}(x_i, y_i, x_{i-1}, y_{i-1}, \dots, x_0, y_0).$$" (Section 2.1. Problem setup)
- "On this controlled testbed, we directly compare two formulations: next-state prediction (regression) and next-token prediction (classification)." (Section 3. Inductive Bias 2: Spatial Stability)
- "Using the first 50 points as context, we autoregressively generate the next 50 points." (Section 3.2. Fair comparison: regression wins over classification)
- "This inspires us to vary the context length to control temporal locality." (Section 1. Introduction)
- Inference: In/Out Dimension are inferred as 3D (x, y, t) because the task is trajectory prediction over 2D positions across time. In/Out Dynamics are inferred as Capped from explicit finite context/horizon use (e.g., first 50 then next 50) and varying but bounded context length. Attention Dynamic is inferred as Static because the model consumes predefined context windows; State Dynamic is inferred as Direct because the objective remains reactive autoregressive prediction.
