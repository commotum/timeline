# Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models (Not specified in the paper.)
Source: Plan-and-Solve Prompting.md

## Core reasons
- Proposes a prompting mechanism that adds explicit planning and step-by-step execution to LLM reasoning, changing how computation is carried out at inference time.
- The method targets missing-step and calculation errors by modifying the reasoning process rather than data, model architecture, or positional encoding.

## Evidence extracts
- "To address the missing-step errors, we propose Planand-Solve (PS) Prompting. It consists of two components: first, devising a plan to divide the entire task into smaller subtasks, and then carrying out the subtasks according to the plan." (Abstract)
- "We introduce PS prompting, a new zero-shot CoT prompting method, which enables LLMs to explicitly devise a plan for solving a given problem and generate the intermediate reasoning process before predicting the final answer for the input problem." (Section 2 Plan-and-Solve Prompting)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
