# Self-Distillation Enables Continual Learning (Not specified in the paper.)
Source: Self-Distillation Enables Continual Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scientific question answering (multiple-choice) | Scientific question text and answer choices | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Final answer choice | 0D (inferred) | Fixed (inferred) |
| Tool-use API call prediction | Tool-API specification and user request tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Tool API call text | 1D (t) (inferred) | Capped (inferred) |
| Medical clinical question answering | Clinical reasoning question tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Medical answer text | 1D (t) (inferred) | Capped (inferred) |
| Knowledge-acquisition factual question answering | Questions about injected 2025 natural-disaster articles | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Factual answer text | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates four text-domain tasks in its continual-learning setup: Science Q&A, Tool Use, Medical clinical reasoning QA, and Knowledge Acquisition QA. Across these tasks, the interfaces are language prompts and generated or selected answers, so the justified task dimensions are 1D (t) text sequences, with one explicit multiple-choice output mapped to 0D. Dynamics are capped by the model interface (context window and max generation length), while attention/state are inferred as static/direct from the fixed autoregressive conditioning and next-token policy formulation.

## Evidence
### Task: Scientific question answering (multiple-choice)
- "- Science Q&A: Undergraduate-level scientific reasoning, using the Chemistry L-3 subset of SciKnowEval (Feng et al., 2024)." (Section 4.1 EXPERIMENTAL SETTING)
- "Since this is a multiple-choice dataset, accuracy was computed by exact match between the model's final answer choice and the ground truth." (Section B.3 DATASET DETAILS)
- Inference: `1D (t)` input and `0D` output are inferred from the question-answer format plus explicit "multiple-choice" and "final answer choice" wording. `Capped` dynamics are inferred from "the full corpus exceeds the model's context window" and the reported "Max generation length ... 2048" for skill learning. `Static` attention and `Direct` state are inferred from the fixed autoregressive interface "the student is simply the base model without this conditioning  $\pi_{\theta}(\cdot|x)$" and on-policy sampling "samples responses from the student policy  $y \sim \pi_{\theta}(\cdot|x)$" (Section 3 SELF-DISTILLATION FINE-TUNING; Section 4.1 EXPERIMENTAL SETTING; Section B.1 TRAINING DETAILS).

### Task: Tool-use API call prediction
- "- *Tool Use*: Mapping a tool-API specification and user request to the correct tool call, using ToolAlpaca (Tang et al., 2023)." (Section 4.1 EXPERIMENTAL SETTING)
- "In this benchmark, the model receives a tool-API specification and a user request, and must identify the correct tool call." (Section 3.2 VALIDATING THE ICL ASSUMPTION)
- Inference: Input/output are inferred as text sequences (`1D (t)`) because the task is specified as mapping language/API descriptions to a generated call, and evaluation checks a textual call format ("regex matching against the ground-truth API call"). `Capped` dynamics, `Static` attention, and `Direct` state are inferred from the same context-window/max-generation and autoregressive-policy evidence above (Section B.3 DATASET DETAILS; Section 3 SELF-DISTILLATION FINE-TUNING; Section B.1 TRAINING DETAILS).

### Task: Medical clinical question answering
- "- *Medical*: Clinical reasoning questions, with training data from stage 1 of the HuatuoGPT-o1 pipeline and evaluation from stage 2 (Chen et al., 2024)." (Section 4.1 EXPERIMENTAL SETTING)
- "Since these are open-ended clinical reasoning questions, we used GPT-5-mini as an automated evaluator" (Section B.3 DATASET DETAILS)
- Inference: Input/output are inferred as text sequences (`1D (t)` to `1D (t)`) from "clinical reasoning questions" and explicitly "open-ended" answer evaluation. `Capped` dynamics, `Static` attention, and `Direct` state are inferred from the same interface evidence: context-window limits, max generation length, and fixed autoregressive conditioning/sampling (Section 4.1 EXPERIMENTAL SETTING; Section B.1 TRAINING DETAILS; Section 3 SELF-DISTILLATION FINE-TUNING).

### Task: Knowledge-acquisition factual question answering
- "In *Knowledge Acquisition*, the objective is different: the model must integrate genuinely new factual content not present in its pretraining data." (Section 4.1 EXPERIMENTAL SETTING)
- "Following Mecklenburg et al. (2024), we generate question–answer pairs about these articles" (Section 4.1 EXPERIMENTAL SETTING)
- "• Out-of-Distribution Accuracy: \"Indirect\" questions whose answers depend on the injected knowledge but do not directly reference it" (Section 4.1 EXPERIMENTAL SETTING)
- Inference: The task is inferred as text QA (`1D (t)` to `1D (t)`) from explicit "question–answer pairs." `Capped` dynamics are supported by context-window and generation-length constraints ("full corpus exceeds the model's context window" and "Max generation length ... 1024" for knowledge acquisition). `Static` attention and `Direct` state are inferred from the same fixed autoregressive policy interface used across tasks (Section 4.1 EXPERIMENTAL SETTING; Table 4 in Section B.1 TRAINING DETAILS; Section 3 SELF-DISTILLATION FINE-TUNING).
