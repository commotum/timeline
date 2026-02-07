# PFformer: A Position-Free Transformer Variant for Extreme-Adaptive Multivariate Time Series Forecasting (Not specified in the paper)
Source: PFformer- A Position-Free Transformer Variant for Extreme-Adaptive Multivariate Time Series Forecasting.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prediction (3-day ahead time series forecasting) | aligned multivariate time series (streamflow and rainfall) | 1D (t) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | future values of target time series (streamflow) over next h steps (3 days) | 1D (t) | Fixed |
| prediction (rolling 4-hour time series forecasting) | aligned multivariate time series (streamflow and rainfall) | 1D (t) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | future values of target time series (streamflow) in rolling 4-hour predictions | 1D (t) | Fixed |

## Summary
The paper focuses on single-target multivariate time series forecasting in a hydrology setting, evaluated in two scenarios: 3-day ahead long-sequence prediction and rolling 4-hour prediction. Inputs and outputs are temporal sequences, so the task operates over 1D (t) data with fixed input windows and fixed output horizons in the described setup. Attention is treated as static over a fixed window and state is constructed via learned embeddings and latent states, based on the model description.

## Evidence
### Task: prediction (3-day ahead time series forecasting)
- "long sequence prediction for 3 days ahead" (Opening paragraph)
- "Each prediction estimates the upcoming 3 days" (Data Descriptions)
- Inference: Inferred Fixed input window from the fixed 15-day history design, Static attention from fixed-window multi-head attention, and Constructed state from the AEE latent-state encoder description. (Data Descriptions; Encoder; Auto-Encoder-Based Embedding)

### Task: prediction (rolling 4-hour time series forecasting)
- "rolling predictions every four hours" (Opening paragraph)
- "predictions made every four hours" (Data Descriptions)
- Inference: Inferred Fixed input window from the fixed 15-day history design, Static attention from fixed-window multi-head attention, and Constructed state from the AEE latent-state encoder description. (Data Descriptions; Encoder; Auto-Encoder-Based Embedding)
