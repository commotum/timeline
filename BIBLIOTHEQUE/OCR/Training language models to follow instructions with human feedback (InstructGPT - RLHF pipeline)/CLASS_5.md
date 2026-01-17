# Training language models to follow instructions with human feedback (Not specified in the paper)
Source: Training language models to follow instructions with human feedback (InstructGPT - RLHF pipeline).md

## Core reasons
- The paper presents an alignment and training methodology based on fine-tuning with human feedback and reinforcement learning, not a new architecture or positional encoding.
- It details a supervised fine-tuning plus reward model plus PPO pipeline to optimize model behavior toward human preferences, which is a foundations/optimization contribution.

## Evidence extracts
- "In this paper, we show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback. Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning. We then collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback." (Abstract)
- "We focus on *fine-tuning* approaches to aligning language models. Specifically, we use reinforcement learning from human feedback (RLHF; Christiano et al., 2017; Stiennon et al., 2020) to fine-tune GPT-3 to follow a broad class of written instructions (see Figure 2)." (Section 1 Introduction)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
