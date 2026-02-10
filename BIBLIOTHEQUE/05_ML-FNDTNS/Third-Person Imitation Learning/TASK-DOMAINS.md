# THIRD-PERSON IMITATION LEARNING (Not specified in the paper)
Source: Third-Person Imitation Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Third-person imitation control (Point) | expert image-based rollouts; novice-domain observations | 3D (x, y, z) or (x, y, t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | policy actions for point reaching | 1D (t) (inferred) | Capped (inferred) |
| Third-person imitation control (Reacher) | expert image-based rollouts; novice-domain observations | 3D (x, y, z) or (x, y, t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | policy actions for arm reaching | 1D (t) (inferred) | Capped (inferred) |
| Third-person imitation control (Inverted Pendulum) | expert image-based rollouts; novice-domain observations | 3D (x, y, z) or (x, y, t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | policy actions for balancing control | 1D (t) (inferred) | Capped (inferred) |
| Expert-vs-non-expert trajectory classification | paired observations/features at t and t+n | 3D (x, y, z) or (x, y, t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | expert/non-expert class probability | 0D (inferred) | Fixed (inferred) |
| Domain-label classification for confusion loss | observation features from D_F(o_t) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | expert-domain/novice-domain class probability | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers third-person imitation control from image-based demonstrations in three MuJoCo tasks: point reaching, reacher reaching, and inverted-pendulum balancing. It also explicitly includes two auxiliary binary classification tasks in the training objective: expert-vs-non-expert discrimination and domain-label prediction for domain confusion. From the OCR text, inputs are visual observations with temporal pairing, supporting 3D (x, y, t) usage (via the glossary’s 3D label) and one single-frame 2D classifier case (both inferred from the stated interfaces). Outputs are temporal action rollouts for control and 0D binary labels for classifiers, with fixed/capped dynamics, static attention, and constructed state inferred from the explicit shared feature extractor design.

## Evidence
### Task: Third-person imitation control (Point)
- "Point: A pointmass attempts to reach a point in a plane." (Section 6.1 Environments)
- "given a collection of expert image-based rollouts in one domain, is it possible to train a policy in a different domain that replicates the essence of the original behavior?" (Section 6 Experiments)
- Inference: Input dimension and dynamics are inferred as spatiotemporal/capped because the method is image-rollout based and finite-horizon ("In third-person learning, observations are more typically available ... we will work with observations  $o_t$" in Section 5.1; "A discrete-time finite-horizon discounted Markov decision process" in Section 3). Output is marked 1D (t) (inferred) because policy training is over rollout time (Algorithm 1 uses per-time-step reward/policy updates).

### Task: Third-person imitation control (Reacher)
- "Reacher: A two DOF arm attempts to reach a designated point in the plane." (Section 6.1 Environments)
- "our proposed algorithm is indeed able to recover reasonable policies for all three tasks we examined." (Section 6.2 Evaluations)
- Inference: The same inferred domain assignments as the point task are supported by shared image-rollout training and finite-horizon control setup (Sections 5.1, 6.1, and Algorithm 1).

### Task: Third-person imitation control (Inverted Pendulum)
- "Inverted Pendulum: A classic RL task wherein a pendulum must be made to balance via control." (Section 6.1 Environments)
- "we do not terminate an episode when the agent falls but rather allow data collection to continue for a fixed horizon." (Section 6.1 Environments)
- Inference: Capped dynamics are inferred directly from the fixed horizon quote; spatiotemporal input and temporal action output are inferred from image rollouts and policy optimization over time (Sections 5.1, 6.1, and Algorithm 1).

### Task: Expert-vs-non-expert trajectory classification
- "the loss in Equation 2 is utilized to train a discriminator  $\mathcal{D}_R$  capable of distinguishing expert vs non-expert policies." (Section 5.1 Game Formulation)
- "The classifier then makes a prediction  $\mathcal{D}_R(\sigma_t,\sigma_{t+n})=\hat{c}_\ell$ ." (Section 5.1 Game Formulation)
- Inference: Input dimension is inferred as spatiotemporal because classification consumes paired time-indexed observations/features (t and t+n); output is inferred 0D as a binary class probability; fixed dynamics/attention are inferred from the fixed pair interface and no runtime input-selection mechanism in the described classifier pathway.

### Task: Domain-label classification for confusion loss
- "The problem is then to ensure that  $D_F$  contains no information regarding the rollout's domain label  $d_\ell$  (i.e., expert vs. novice domain)." (Section 5.1 Game Formulation)
- "another classifier  $\mathcal{D}_D$ , which takes features produced by  $D_F$  and outputs the probability that those features were produced by in the expert vs. non-expert environment." (Section 5.1 Game Formulation)
- Inference: The 2D input assignment is inferred because features come from image observations ("Input is images are size 50 x 50 with 3 channels, RGB." in Appendix B), while 0D output and fixed/static interface are inferred from binary domain probability prediction on a fixed-size feature vector pathway.
