# Neural Ordinary Differential Equations (Year not specified)
Source: Neural Ordinary Differential Equations (Neural ODEs).md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract defines the core method as continuous-depth ODE dynamics solved with a differential equation solver, not Transformer/self-attention blocks.
- Auxiliary analyses mark attention signals as not specified and describe ODE-Net/CNF/latent ODE model families; the extending-dimensions analysis file was unavailable (`MISSING`).

## Evidence
- "Instead of specifying a discrete sequence of hidden layers, we parameterize the derivative of the hidden state using a neural network." (Abstract, `Neural Ordinary Differential Equations (Neural ODEs).md`)
- "The output of the network is computed using a blackbox differential equation solver." (Abstract, `Neural Ordinary Differential Equations (Neural ODEs).md`)
- "Attention dynamics are not specified" (Summary, `TASK-DOMAINS.md`)
- "...,Not specified in the paper.,..." in the `attention_dynamic` field across tasks. (`TASK-DOMAINS.csv`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for high-confidence TRANSFORMER-NO.
Pass 2 (targeted source scan): skipped - not needed because Pass 1 was decisive.
