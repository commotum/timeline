# The Surprising Effectiveness of Test-Time Training for Few-Shot Learning (Not specified in the paper.)
Source: The Surprising Effectiveness of Test-Time Training for Few-Shot Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prediction (ARC visual puzzle transformation) | 2D color-grid demonstration pairs and a test input grid | 2D (x, y) | Capped | Static (inferred) | Constructed (inferred) | Predicted 2D output grid(s) | 2D (x, y) | Capped |
| Prediction (BBH natural-language reasoning answers) | Natural-language demonstration pairs and a query input | 1D (t) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Natural-language answer tokens/labels | 1D (t) | Capped (inferred) |

## Summary
The paper evaluates one TTT-enabled language-model framework on two distinct task domains: ARC visual puzzle transformation and BBH natural-language reasoning/answer prediction. ARC is explicitly 2D grid-to-grid mapping with bounded grid size and bounded demo/test counts, supporting 2D (x, y) and Capped dynamics. BBH is natural-language task solving over text sequences, supporting 1D (t), with Capped dynamics marked as inferred from the fixed 10-shot setup and finite decoding context. Across both domains, Attention is Static (inferred) and State is Constructed (inferred) because the model consumes fixed prompt context while building temporary task-specific parameter state via test-time updates.

## Evidence
### Task: Prediction (ARC visual puzzle transformation)
- "Each puzzle (henceforth referred to as a task) consists of input-output pairs of 2D grids (up to  $30 \times 30$  in size) containing shapes or patterns in up to 10 different colors, as displayed in Figure 3." (Section 4.1. Background)
- "The output of each pair is obtained by applying an *intuitive* and *shared* transformation or rule y = f(x). Each task has 2-7 demonstration examples and 1-3 test examples." (Section 4.1. Background)
- "Test-time training (TTT) enables parametric models to adapt during inference through dynamic parameter updates in response to each test input." (Section 2.2. Test-Time Training)
- Inference: Attention is labeled Static because the model is conditioned on predefined in-context demonstrations and test input (Sections 2.1 and 3.1) rather than runtime selection of new observations. State is labeled Constructed because TTT creates temporary adapted parameters ("resulting in temporarily updated parameters  $\theta_d$ , which are subsequently used for prediction."; Section 2.2).

### Task: Prediction (BBH natural-language reasoning answers)
- "BIG-Bench Hard (BBH; Srivastava et al., 2023; Suzgun et al., 2023) is a benchmark comprising 27 challenging tasks across 23 task types, designed to evaluate large language models on reasoning, compositionality, and generalization." (Section 5.1. Background)
- "Unlike ARC, BBH features a broader natural language structure and lacks a shared input format, making it unsuitable for invertible transformations." (Section 5.1. Background)
- "For the 27 tasks in BBH, we consider the 10shot setting, where we select 10 random pairs from each task's dataset to be demonstration pairs and evaluate on the remaining data." (Section 5.2. Experimental Details)
- "For each task d, we train a separate set of LoRA parameters at test-time, with a LoRA rank of 64 over 40 random shuffles of the demonstration pairs to produce leave-one-out in-context tasks." (Section 5.2. Experimental Details)
- Inference: In Dimension is 1D (t) from the paper’s natural-language task format. In/Out Dynamics are labeled Capped (inferred) from fixed 10-shot prompting and finite answer generation. Attention is labeled Static (inferred) because prompts are predefined, and State is Constructed (inferred) because task-specific LoRA parameters are updated at test time.
