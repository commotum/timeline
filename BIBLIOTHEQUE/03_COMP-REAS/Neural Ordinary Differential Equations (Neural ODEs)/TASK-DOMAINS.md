# Neural Ordinary Differential Equations (Not specified in the paper)
Source: Neural Ordinary Differential Equations (Neural ODEs).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification (supervised learning) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| density estimation (maximum likelihood) | data samples / data points | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | density / log-likelihood values | 0D (inferred) | Not specified in the paper. |
| generation (sampling from learned density) | latent samples from an initial distribution | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | generated data samples | Not specified in the paper. | Not specified in the paper. |
| time-series modeling / prediction | time-series observations with timestamps | 1D (t) (inferred) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | reconstructed / predicted time-series observations at arbitrary time points | 1D (t) (inferred) | Open (inferred) |
| event time modeling (Poisson process likelihood) | observation/event times | 1D (t) (inferred) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | event rate / likelihood over time | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper reports supervised learning (classification), density estimation with continuous normalizing flows (including sampling), and continuous-time time-series modeling, including modeling observation times with a Poisson process likelihood. Temporal tasks are supported with time-indexed observations and predictions, which justify 1D (t) structure and open dynamics (inferred) due to arbitrary observation and prediction times. Attention dynamics are not specified, while the latent ODE time-series models introduce constructed latent trajectories (inferred) as state.

## Evidence
### Task: classification (supervised learning)
- "Replacing residual networks with ODEs for supervised learning" (Section 3)
- "In this section, we experimentally investigate the training of neural ODEs for supervised learning." (Section 3)
- "In the classification and density estimation experiments, we were able to reduce the tolerance to 1e-3 and 1e-5, respectively, without degrading performance." (Section 6)

### Task: density estimation (maximum likelihood)
- "We also construct continuous normalizing flows, a generative model that can train by maximum likelihood" (Abstract)
- "train the flow on a density estimation task by performing maximum likelihood estimation" (Section 4.1)
- "maximizes  $\mathbb{E}_{p(\mathbf{x})}[\log q(\mathbf{x})]$" (Section 4.1)
- Inference: Out Dimension set to 0D because the output is a scalar log-density value ("log q(\mathbf{x})"). (Section 4.1)

### Task: generation (sampling from learned density)
- "then afterwards reverse the CNF to generate random samples from  $q(\mathbf{x})$ ." (Section 4.1)
- "We first compare continuous and discrete planar flows at learning to sample from a known distribution." (Section 4.1)

### Task: time-series modeling / prediction
- "We present a continuous-time, generative approach to modeling time series." (Section 5)
- "Using ODEs as a generative model allows us to make predictions for arbitrary time points  $t_1...t_M$  on a continuous timeline." (Training and Prediction)
- "We investigate the ability of the latent ODE model to fit and extrapolate time series." (Section 5.1)
- Inference: In/Out Dimension set to 1D (t) and In/Out Dynamics set to Open because observations and predictions are indexed by time and can occur at arbitrary time points. State Dynamic set to Constructed because the model represents each series by a latent trajectory. (Section 5; Training and Prediction)

### Task: event time modeling (Poisson process likelihood)
- "The rate of events can be parameterized by a function of the latent state:  $p(\text{event at time }t|\mathbf{z}(t)) = \lambda(\mathbf{z}(t))$ ." (Poisson Process likelihoods)
- "the likelihood of a set of independent observation times in the interval  $[t_{\text{start}}, t_{\text{end}}]$  is given by an inhomogeneous Poisson process" (Poisson Process likelihoods)
- "A Poisson process likelihood on observation times can be combined with a data likelihood" (Poisson Process likelihoods)
- Inference: In/Out Dimension set to 1D (t) and In/Out Dynamics set to Open because the likelihood is defined over observation times across an interval. State Dynamic set to Constructed because the event rate depends on the latent state. (Poisson Process likelihoods)
