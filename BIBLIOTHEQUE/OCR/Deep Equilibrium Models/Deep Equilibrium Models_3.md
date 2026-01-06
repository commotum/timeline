# Deep Equilibrium Models (Not specified in the paper.)
Source: Deep Equilibrium Models.md

## Core reasons
- The paper proposes a new computation mechanism for sequence models by directly solving for equilibrium (fixed-point) states via root-finding instead of layer-by-layer feedforward depth.
- It reframes inference and training around implicit differentiation through equilibrium points, emphasizing constant-memory computation rather than architectural changes like positional encoding or dimensional lifting.

## Evidence extracts
- "We present a new approach to modeling sequential data: the deep equilibrium model (DEQ). Motivated by an observation that the hidden layers of many existing deep sequence models converge towards some fixed point, we propose the DEQ approach that directly finds these equilibrium points via root-finding." (Abstract)
- "This solution corresponds to the eventual hidden layer values of an *infinite depth* network. But instead of finding this value by iterating the model, we propose to directly (and in practice, more quickly) solve for the equilibrium via any black-box root-finding method." (1 Introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
