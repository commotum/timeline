# REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS (Not specified in the paper.)
Source: ReAct- Synergizing Reasoning and Acting in Language Models.md

## Core reasons
- Proposes a new inference paradigm that interleaves reasoning traces with actions, changing how computation proceeds during task solving.
- Defines a mechanism that augments the agent action space with language "thoughts" to update context and guide subsequent reasoning and acting.

## Evidence extracts
- "In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with and gather additional information from external sources such as knowledge bases or environments." (Abstract)
- "The idea of ReAct is simple: we augment the agent's action space to  $\hat{A} = A \cup \mathcal{L}$ , where  $\mathcal{L}$  is the space of language. An action  $\hat{a}_t \in \mathcal{L}$  in the language space, which we will refer to as a *thought* or a *reasoning trace*, does not affect the external environment, thus leading to no observation feedback. Instead, a thought  $\hat{a}_t$  aims to compose useful information by reasoning over the current context  $c_t$ , and update the context  $c_{t+1} = (c_t, \hat{a}_t)$  to support future reasoning or acting." (Section 2)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
