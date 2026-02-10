# Reward learning from human preferences and demonstrations in Atari (Not specified in the paper)
Source: Reward learning from human preferences and demonstrations in Atari.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sequential control in Atari from demonstrations and learned reward | observation stream from Atari environment (stacked frames) | 1D (t); 3D (x, y, t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | action sequence | 1D (t) (inferred) | Open (inferred) |
| Reward estimation from observations | observation frames | 3D (x, y, t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | scalar reward estimate (r_{t+1} in R) | 0D (inferred) | Fixed (inferred) |
| Pairwise preference prediction over trajectory clips | pair of trajectory clips (25 agent steps each) | 1D (t); 3D (x, y, t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | preference label/probability over clip pair | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers a reinforcement learning control task in Atari and two reward-learning tasks that map observations and clip pairs to reward-related supervision. Inputs are spatiotemporal Atari observations (stacked frames) and short trajectory clips, with outputs as action sequences, scalar rewards, and pairwise preference judgments. Dimension coverage is primarily 3D (x, y, t) inputs plus 1D temporal sequencing, with 0D scalar/label outputs for reward and preference tasks. Based on the described pipeline, control operates with Open interaction dynamics, while reward and preference modeling use fixed-size clip/frame interfaces; attention and state are inferred as Static and Direct.

## Evidence
### Task: Sequential control in Atari from demonstrations and learned reward
- "in time step t the agent receives an observation  $o_t$  from the environment and takes an action  $a_t$ ." (Section 2.1 Setting)
- "The goal of the agent is to approximate as closely as possible the behavior intended by the human. It achieves this by 1. imitating the behavior from the demonstrations, and 2. attempting to maximize a reward function inferred from the preferences and demonstrations." (Section 2.1 Setting)
- Inference: In Dimension includes 3D (x, y, t) from stacked observations ("frame stacking of 4 frames" and "we treat 4 frames as one observation"). Open dynamics is inferred from ongoing interaction ("interacting sequentially with an environment over a number of time steps") and continuous episodes ("effectively converting the environment into a single continuous episode"). Attention Dynamic is inferred as Static and State Dynamic as Direct from fixed observation input to Q-values and action selection ("outputs a set of action-values  $Q(o, \cdot; \theta)$  for a given input observation o" and "The agent's behavior is  $\epsilon$ -greedy with respect to the action-value function  $Q(o, \cdot; \theta)$ ."). (Section A.1 Environment; Section 2.1 Setting; Section 2.3 Training the policy)

### Task: Reward estimation from observations
- "Our reward model is a convolutional neural network  $\hat{r}$  taking observation  $o_t$  as input (we omit actions in our experiments) and outputting an estimate of the corresponding reward  $r_{t+1} \in \mathbb{R}$ ." (Section 2.4 Training the reward model)
- "For the reward model, we use the same configuration as the Atari experiments in Christiano et al. (2017): 84x84x4 stacked frames (same as the inputs to the policy) as inputs to 4 convolutional layers ... This is followed by a fully connected layer of size 64 and then a scalar output." (Section A.3 Agent and reward model)
- Inference: 3D (x, y, t) and Fixed input dynamics are inferred from fixed 84x84x4 stacked-frame inputs; 0D and Fixed output dynamics are inferred from the scalar reward output. Attention Dynamic is inferred as Static and State Dynamic as Direct because the paper specifies direct observation-to-reward mapping without a separate runtime memory/state construction mechanism in the reward model description. (Section 2.4 Training the reward model; Section A.3 Agent and reward model)

### Task: Pairwise preference prediction over trajectory clips
- "The annotator is given a pair of clips, which are trajectory segments of 25 agent steps each (approximately 1.7 seconds long). The annotator then indicates which clip is preferred, that the two clips are equally preferred, or that the clips cannot be compared." (Section 2.4 Training the reward model)
- "To train the reward model  $\hat{r}$  on preferences, we interpret the reward model as a preference predictor" (Section 2.4 Training the reward model)
- Inference: Input dimension is inferred as 1D (t); 3D (x, y, t) because each sample is a time-indexed clip built from stacked-frame observations; Fixed dynamics are inferred from fixed clip length ("each of 25 actor steps"). Output is treated as 0D Fixed because preference supervision is a bounded per-pair label/probability ("the judgment label (one of (0,1), (1,0) or (0.5,0.5))"). Attention Dynamic is inferred as Static and State Dynamic as Direct from the fixed clip-pair comparison formulation. (Section 2.4 Training the reward model; Section A.3 Agent and reward model)
