# Neural Ordinary Differential Equations (Not specified in the paper.)
Source: Neural Ordinary Differential Equations (Neural ODEs).md

## Core reasons
- Proposes replacing discrete layers with continuous-depth dynamics defined by an ODE, changing how computation proceeds through the model.
- Emphasizes adaptive computation and constant-memory training via an ODE solver and adjoint method, indicating a new computation mechanism rather than a dataset or positional encoding change.

## Evidence extracts
- "Instead of specifying a discrete sequence of hidden layers, we parameterize the derivative of the hidden state using a neural network. The output of the network is computed using a blackbox differential equation solver. These continuous-depth models have constant memory cost, adapt their evaluation strategy to each input" (Abstract)
- "In the limit, we parameterize the continuous dynamics of hidden units using an ordinary differential equation (ODE) specified by a neural network" (Section 1 Introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
