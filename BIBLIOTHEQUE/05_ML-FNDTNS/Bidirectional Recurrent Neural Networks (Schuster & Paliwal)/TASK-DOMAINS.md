# Bidirectional Recurrent Neural Networks (1997)
Source: Bidirectional Recurrent Neural Networks (Schuster & Paliwal).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Regression (unimodal) | Time sequence of input data vectors | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | Time sequence of continuous output vectors | 1D (t) (inferred) | Not specified in the paper. |
| Classification (per-time-step) | Time sequence of input data vectors | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | Time sequence of class labels/probabilities | 1D (t) (inferred) | Not specified in the paper. |
| Sequence posterior probability estimation | Time sequence of input vectors plus class-label context (c_1...c_{t-1}) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | Conditional probability of a complete class sequence | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers temporal sequence tasks including unimodal regression and per-time-step classification (including phoneme classification), and it also targets sequence posterior probability estimation for class sequences. Inputs and outputs are described as time sequences of vectors or labels, supporting 1D (t) dimensions, while the sequence-probability task yields a single scalar probability. The recurrent architectures use explicit state neurons and are described as using all past and future input information, supporting constructed state and static attention (both inferred). The paper does not specify explicit interface limits on sequence length, so dynamics are left unspecified.

## Evidence
### Task: Regression (unimodal)
- "Consider a (time) sequence of input data vectors" (Section I.B Technical)
- "and a sequence of corresponding output data vectors" (Section I.B Technical)
- "When outputs are continuous, the problem is known as a regression problem" (Section I.B Technical)
- "Unimodal regression (i.e., compute  $\hat{\mathbf{y}}_t = E[\mathbf{y}_t|\mathbf{x}_1^T]$ )" (Section I.C)
- "input information in the past and the future of the currently evaluated time frame can directly be used" (Section II.B)
- "split the state neurons of a regular RNN in a part that is responsible for the positive time direction (forward states)" (Section II.B)
- Inference: 1D (t) input/output because the paper defines time sequences; Static attention from the statement that past and future input information can be used; Constructed state from the use of state neurons (Sections I.B, II.B).

### Task: Classification (per-time-step)
- "one seeks the most probable class out of a given pool of K classes for every time frame t" (Section I.B Technical)
- "given an input vector sequence  $\mathbf{x}_1^T$" (Section I.B Technical)
- "classification [i.e., compute  $\hat{y}_t^{(k)} = \Pr(C_t = k|\mathbf{x}_1^T)$" (Section I.C)
- "classification of phonemes from the TIMIT speech database." (Section II.C)
- "input information in the past and the future of the currently evaluated time frame can directly be used" (Section II.B)
- "split the state neurons of a regular RNN in a part that is responsible for the positive time direction (forward states)" (Section II.B)
- Inference: 1D (t) input/output because the paper defines time sequences; Static attention from the statement that past and future input information can be used; Constructed state from the use of state neurons (Sections I.B, II.B).

### Task: Sequence posterior probability estimation
- "Estimation of the conditional probability of a complete sequence of classes of length T" (Section I.C)
- "compute  $\Pr(c_1, c_2, \dots, c_T | \mathbf{x}_1^T)$" (Section I.C)
- "goal is to train a network to estimate conditional probabilities of the kind  $\Pr(c_t|c_1, c_2, \dots, c_{t-1}, \mathbf{x}_1^T)$" (Section III.A)
- "conditioned on continuous  $(\mathbf{x}_1^T)$  and discrete inputs  $(c_1, c_2, \dots, c_{t-1})$" (Section III.B)
- "input information in the past and the future of the currently evaluated time frame can directly be used" (Section II.B)
- "split the state neurons of a regular RNN in a part that is responsible for the positive time direction (forward states)" (Section II.B)
- Inference: 1D (t) input from the time-sequence formulation; 0D output and Fixed dynamics because the task is a single sequence probability; Static attention from the statement that past and future input information can be used; Constructed state from the use of state neurons (Sections I.C, II.B).
