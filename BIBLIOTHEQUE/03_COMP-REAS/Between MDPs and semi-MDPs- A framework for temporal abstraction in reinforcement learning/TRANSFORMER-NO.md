# Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning (1999)
Source: TASK-DOMAINS.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: hint-only

## Why
- Hint summaries and evidence describe classic reinforcement learning with options/SMDP Q-learning in gridworld and control tasks, not neural self-attention architectures.
- No Transformer/attention-block terminology appears in the hint files; "attention" appears only as an inferred task-table attribute, not as a model mechanism.

## Evidence
- "As an illustration, we applied SMDP Q-learning to the rooms example with the goal at  $G_1$  and at  $G_2$  (Fig. 2). As in the case of planning, we used three different sets of options,  $\\mathcal{A}, \\mathcal{H}$ , and  $\\mathcal{A} \\cup \\mathcal{H}$ ." (TASK_MODEL_RATIO.md, item 2)
- "Options consist of three components: a policy  $\\pi: \\mathcal{S} \\times \\mathcal{A} \\to [0, 1]$ , a termination condition  $\\beta: \\mathcal{S}^+ \\to [0, 1]$ , and an initiation set  $\\mathcal{I} \\subseteq \\mathcal{S}$ ." (TASK-DOMAINS.md, Evidence section)

## Pass accounting
Pass 0 (hint-first): performed - decisive evidence of options/SMDP RL methods and no Transformer architecture cues.
Pass 1 (source triage): skipped - hint files were sufficient for a high-confidence NO decision.
Pass 2 (source deep dive): skipped - not needed after decisive hint-only triage.
