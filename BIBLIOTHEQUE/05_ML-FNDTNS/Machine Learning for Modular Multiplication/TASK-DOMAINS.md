# Machine Learning for Modular Multiplication (Not specified in the paper.)
Source: Machine Learning for Modular Multiplication.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Secret recovery (1D LWE) | Dataset of integer pairs (a_i, b_i) with noise | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Secret s (integer) | 0D (inferred) | Fixed (inferred) |
| Sequence-to-sequence modular multiplication | Digit token sequence representing a_i (base B) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Digit token sequence representing b_i (base B) | 1D (t) (inferred) | Fixed (inferred) |

## Summary
The paper studies two machine learning tasks around modular multiplication: circular regression for recovering a secret s from datasets of integer pairs, and a transformer that maps digit token sequences of a_i to digit token sequences of b_i in base B. The inputs are discrete numeric datasets or token sequences (classified as 1D), and outputs are either a single integer secret (0D) or a fixed-length token sequence (1D). The circular regression procedure maintains an iterative internal guess, while the transformer is framed as a standard sequence-to-sequence mapping with fixed-length inputs and outputs.

## Evidence
### Task: Secret recovery (1D LWE)
- "Given a data set consisting of pairs of integers  $\{(a_i,b_i)\}_{1\leq i\leq m}$ , where  $b_i\equiv a_is+e_i\pmod{p}$  and the 'noise' or 'error' values  $e_i$  are sampled from a centered discrete Gaussian distribution with standard deviation  $\sigma$ , the task is to find the unknown secret s." (Section 2.1)
- "So we start with an initial guess for s, call it  $s_0$ , and at each time step t we define

$$s_{t+1} = s_t + \eta \frac{2\pi}{p} \sum_{i=1}^{m} a_i \sin\left(y_i - \frac{2\pi}{p} a_i s_t\right),$$" (Section 2.2)
- Inference: Classified the input as 1D (t) and Fixed because the dataset is indexed by i = 1..m; attention as Static because the algorithm aggregates over a fixed dataset/batch rather than selecting inputs at runtime; state as Constructed because it maintains an iterative guess s_t; output as 0D Fixed because the task returns a single integer secret. (Section 2.1, Section 2.2)

### Task: Sequence-to-sequence modular multiplication
- "We now move on to an alternative machine learning-based approach to modular multiplication, namely the use of *transformers*, which are a class of deep learning models designed for \"sequence-to-sequence\" tasks: transforming one sequence of elements (e.g. words) to another." (Section 3)
- "we represent the integer  $a_i$  as an input sequence of t tokens  $x_{i,1}...x_{i,t}$  in a given base  $\mathcal{B}$ , and train a transformer-based  $\mathcal{M}$  to output  $b_i$  represented as an output sequence of t tokens  $y_{i,1}...y_{i,t}$  in the same base  $\mathcal{B}$ ." (Section 3.1)
- Inference: Classified inputs/outputs as 1D (t) and Fixed because they are sequences of t tokens; attention as Static based on a standard sequence-to-sequence transformer operating over the full input sequence; state as Direct because the model is described as a standard seq2seq mapping without explicit constructed external state. (Section 3, Section 3.1)
