# DriveWorld: 4D Pre-trained Scene Understanding via World Models for Autonomous Driving (2024)
Source: a365eb-2024.pdf

## Core reasons
- DriveWorld introduces a world model-based 4D spatio-temporal pre-training framework that learns from multi-camera driving videos to build representations usable across perception, prediction, and planning tasks.
- The Memory State-Space Model (Dynamic Memory Bank, Static Scene Propagation, and Task Prompt) reshapes the computation pipeline so the model can capture temporal dynamics, spatial statics, and task-aware features before decoding future states.

## Evidence extracts
- "In this paper, we address this challenge by introducing a world model-based autonomous driving 4D representation learning framework, dubbed DriveWorld, which is capable of pre-training from multi-camera driving videos in a spatio-temporal fashion." (Abstract)
- "In this work, we explore 4D pre-training via world models to deal with both aleatoric and epistemic uncertainties. Specifically, we design the Memory State-Space Model to reduce uncertainty within autonomous driving from two aspects: the Dynamic Memory Bank module for learning temporal-aware latent dynamics to predict future states, the Static Scene Propagation module for learning spatial-aware latent statics to provide comprehensive scene context, and the Task Prompt to tune the feature extraction network adaptively for different driving downstream tasks." (Section 1)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
