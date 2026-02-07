# Muesli: Combining Improvements in Policy Optimization (2021)
Source: Muesli- Combining Improvements in Policy Optimization.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control (policy optimization in episodic MDPs) | states $S_t$ (environment states over time) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | actions $A_t$ / policy $\pi$ | 1D (t) (inferred) | Not specified in the paper. |
| Prediction of rewards, values, and policies (learned model) | initial state $s_t$ and action sequence $a_{< t+k}$ | 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Constructed (inferred) | predicted rewards $\hat{r}_k$, values $\hat{v}_k$, policies $\hat{\pi}_k$ | 1D (t) (inferred) | Fixed (inferred) |

## Summary
Muesli is framed as a reinforcement-learning control method for episodic MDPs, evaluated on Atari, 9x9 Go self-play, and MuJoCo continuous-control environments. The interaction is described in time steps, so the input/output are temporal streams (1D (t), inferred), but the paper does not specify interface size dynamics for the control task. The method also trains a learned model that predicts rewards, values, and policies from state/action sequences, using constructed internal representations (inferred) and fixed-length unrolls (inferred).

## Evidence
### Task: Control (policy optimization in episodic MDPs)
- "The agent starts at a state  $S_0 \sim \mu$  from the initial state distribution." (Section 2. Background)
- "At each time step t, the agent takes an action  $A_t \sim \pi(A_t|S_t)$  from a policy  $\pi$" (Section 2. Background)
- "The majority of our experiments were performed on 57 classic Atari games from the Arcade Learning Environment" (Section 1. Introduction)
- "we performed experiments on a suite of continuous control environments (based on MuJoCo and sourced from the OpenAI Gym" (Section 1. Introduction)
- "We also conducted experiments in 9x9 Go in self-play" (Section 1. Introduction)
- "conditioning the model not on a raw environment state  $s_t$  but, instead, on the activations  $h(s_t)$  from a hidden layer of the policy network." (Section 4.4. Using the model)
- Inference: In/Out Dimension set to 1D (t) because the task is defined over time steps; State Dynamic set to Constructed because the system uses hidden-layer representations for modeling. (Sections 2 and 4.4)

### Task: Prediction of rewards, values, and policies (learned model)
- "our model is not trained to reconstruct observations, but is rather only required to provide accurate estimates of rewards, values and policies." (Section 4.3. Learning a model)
- "For training, the model is unrolled k > 1 steps, taking as inputs an initial state  $s_t$  and an action sequence  $a_{< t+k}$ ." (Section 4.3. Learning a model)
- "On each step the model then predicts rewards  $\hat{r}_k$ , values  $\hat{v}_k$  and policies  $\hat{\pi}_k$ ." (Section 4.3. Learning a model)
- "conditioning the model not on a raw environment state  $s_t$  but, instead, on the activations  $h(s_t)$  from a hidden layer of the policy network." (Section 4.4. Using the model)
- Inference: In/Out Dimension set to 1D (t) and In/Out Dynamics set to Fixed because the model is unrolled for a fixed number of steps k; State Dynamic set to Constructed due to hidden-layer conditioning. (Sections 4.3 and 4.4)
