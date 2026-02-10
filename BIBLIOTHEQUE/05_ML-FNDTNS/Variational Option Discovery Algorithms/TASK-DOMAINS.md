# Variational Option Discovery Algorithms (Not specified in the paper)
Source: Variational Option Discovery Algorithms.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unsupervised option discovery for context-conditioned embodied control | state observations s_t and sampled context c | 0D; 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | action-conditioned behavior trajectories | 1D (t) (inferred) | Capped (inferred) |
| Context decoding (trajectory-to-context classification) | trajectories tau (VALOR) or state-based trajectory slices (VIC/DIAYN) | 0D; 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | context prediction P_D(c|.) | 0D (inferred) | Capped (inferred) |
| Downstream hierarchical control in Ant-Maze | Ant-Maze state stream with fixed pretrained VALOR lower level | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | upper-level control actions through a two-level hierarchy | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper covers two core intents in one framework: context-conditioned embodied control and context recovery from behavior, then tests transfer to downstream Ant-Maze control. The described data domains are temporal trajectories and state streams (1D (t), inferred), with discrete context codes as point-like outputs/conditions (0D, inferred). Where the paper gives explicit bounds (trajectory lengths such as T=250/T=1000 and context growth to K_max), dynamics are Capped; several downstream interface limits are not explicitly specified. Recurrent policy/decoder and hierarchical-controller descriptions support Constructed state dynamics, while the core setup supports Static attention because the processed observations are preselected by design.

## Evidence
### Task: Unsupervised option discovery for context-conditioned embodied control
- "Our aim is to learn a policy  $\pi$  where action distributions are conditioned on both the current state  $s_t$  and a *context* c which is sampled at the start of an episode and kept fixed throughout." (Section 3 Variational Option Discovery Algorithms)
- "A context c is sampled from a noise distribution G, and then encoded into a trajectory  $\tau = (s_0, a_0, ..., s_T)$  by a policy  $\pi(\cdot|s_t, c)$ ; afterwards c is decoded from  $\tau$  with a probabilistic decoder D." (Section 3 Variational Option Discovery Algorithms)
- "Scores are evaluated on trajectories of length T=1000 steps, even though agents are trained on trajectories with T=250; we find that using longer horizons at test time clarifies the differences between behaviors." (Section D.1 Guide to Reading This Section)
- Inference: `0D; 1D (t)`, `Capped`, `Static`, and `Constructed` are inferred from sequential trajectory definitions (`\tau = (s_0, a_0, ..., s_T)`), explicit finite trajectory lengths, and "Also unlike prior work, we use recurrent neural network policy architectures." (Section 4 Experimental Setup).

### Task: Context decoding (trajectory-to-context classification)
- "afterwards c is decoded from  $\tau$  with a probabilistic decoder D." (Section 3 Variational Option Discovery Algorithms)
- "Update decoder by supervised learning to maximize  $E[\log P_D(c|\tau)]$ , using batch  $\mathcal{D}$" (Algorithm 1, Section 3 Variational Option Discovery Algorithms)
- "The standard approach for context distributions, used in VIC and DIAYN, is to have K discrete contexts with a uniform distribution:  $c \sim \text{Uniform}(K)$ ." (Section 3.3 Curriculum Approach)
- "$K \leftarrow \min\left(\operatorname{int}\left(1.5 \times K + 1\right), K_{max}\right)," (Section 3.3 Curriculum Approach)
- Inference: `0D; 1D (t)` and `Capped` follow from decoding a single context code from trajectory/state sequences with bounded K; `Static` and `Constructed` are inferred from "We implement VALOR with a recurrent architecture for the decoder (Fig. 1), using a bidirectional LSTM to make sure that both the beginning and end of a trajectory are equally important." and "We only use N=11 equally spaced observations from the trajectory as inputs" (Section 3.2 VALOR).

### Task: Downstream hierarchical control in Ant-Maze
- "- Are the learned behaviors useful for downstream control tasks?" (Section 4 Experimental Setup)
- "Downstream Tasks: We investigated whether behaviors learned by variational option discovery could be used for a downstream task by taking a policy trained with VALOR on the Ant robot (Uniform distribution, seed 10; see Appendix D.7), and using it as the lower level of a two-level hierarchical policy in Ant-Maze." (Section 5 Results, Downstream Tasks)
- "We held the VALOR policy fixed throughout downstream training, and only trained the upper level policy, using A2C as the RL algorithm (with reinforcement occuring only at the lower level—the upper level actions were trained by signals backpropagated through the lower level)." (Section 5 Results, Downstream Tasks)
- Inference: `1D (t)` and `Constructed` are inferred from sequential RL control and explicit two-level hierarchy; `In Dynamics`, `Attention Dynamic`, and `Out Dynamics` remain `Not specified in the paper.` because the downstream section does not state those interface constraints directly.
