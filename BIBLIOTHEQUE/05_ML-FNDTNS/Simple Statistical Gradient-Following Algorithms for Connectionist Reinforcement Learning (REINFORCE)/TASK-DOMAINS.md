# Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (Not specified in the paper.)
Source: Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (REINFORCE).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Associative immediate-reinforcement input-output mapping | External input patterns from the environment | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Network output activity/actions | 0D (inferred) | Fixed (inferred) |
| Episodic delayed-reinforcement sequence learning (temporal credit assignment) | Time-indexed non-reinforcement input over episode steps | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Time-indexed network outputs/actions optimized by episodic reinforcement | 1D (t) (inferred) | Capped (inferred) |
| Nonassociative function optimization | No nonreinforcement input; scalar reinforcement/objective feedback | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Sampled scalar output/action value (e.g., y) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers reinforcement-learning task intents centered on associative immediate input-output mapping, an episodic extension for delayed credit assignment, and nonassociative function optimization. From the OCR evidence, task structure spans point-like per-trial/scalar interactions (0D) and time-indexed episodic sequences (1D (t)). Dynamics are fixed at per-step interfaces and capped for finite-horizon episodes. Attention is static throughout, while state is direct in immediate/nonassociative settings and constructed in recurrent episodic settings.

## Evidence
### Task: Associative immediate-reinforcement input-output mapping
- "In this article we present analytical results concerning certain algorithms for tasks that are associative, meaning that the learner is required to perform an input-output mapping, and, with one limited exception, that involve immediate reinforcement, meaning that the reinforcement (i.e., payoff) provided to the learner is determined by the most recent input-output pair only." (Section 1. Introduction)
- "The network operates by receiving external input from the environment, propagating the corresponding activity through the net, and sending the activity produced at its output units to the environment for evaluation. The evaluation consists of the scalar reinforcement signal r" (Section 2. Reinforcement-learning connectionist networks)
- Inference: 0D/Fixed/Static/Direct are inferred because this setting is described as per-trial input-output pairs in a fixed feedforward network interface ("input ... is a vector" tied to network connectivity), with no runtime observation-selection mechanism or explicit persistent constructed memory described (Section 2. Reinforcement-learning connectionist networks).

### Task: Episodic delayed-reinforcement sequence learning (temporal credit assignment)
- "Now we consider how the REINFORCE class of algorithms can be extended to certain learning problems having a temporal credit-assignment component, as may occur when the network contains loops or the environment delivers reinforcement values with unknown, possibly variable, delays." (Section 5. Episodic REINFORCE algorithms)
- "assume a net N is trained on an episode-by-episode basis, where each episode consists of k time steps, during which the units may recompute their outputs and the environment may alter its non-reinforcement input to the system at each time step. A single reinforcement value r is delivered to the net at the end of each episode." (Section 5. Episodic REINFORCE algorithms)
- "In the case of the recurrent networks, the objective was to learn a trajectory and episodic REINFORCE was used." (Section 8.1. Convergence properties)
- Inference: 1D (t) is inferred from explicit time-step indexing over episodes; Capped is inferred from finite episode length k; Constructed state is inferred from looped/recurrent operation plus accumulation of temporal eligibilities ("single accumulator for each parameter") (Section 5. Episodic REINFORCE algorithms).

### Task: Nonassociative function optimization
- "Consider first a Bernoulli unit having no (nonreinforcement) input and suppose that the parameter to be adapted is p_i = Pr{y_i = 1}." (Section 4. REINFORCE algorithms)
- "Williams and Peng (1991) have also investigated a number of variants of REINFORCE in nonassociative function-optimization tasks, using networks of Bernoulli units." (Section 8.1. Convergence properties)
- "For the Gaussian unit studies mentioned above, the problems considered were nonassociative, involving optimization of a function of a single real variable y, and the adaptable parameters were taken to be μ and σ." (Section 8.2. Gaussian unit search behavior)
- Inference: 0D/Fixed/Static/Direct are inferred from the scalar, nonassociative setup (single variable y; no nonreinforcement input) and the absence of any runtime attention-selection or explicit constructed task-state mechanism in this formulation (Sections 4, 8.1, 8.2).
