# Test-time Adaptation of Tiny Recursive Models (2025)
Source: Test-time Adaptation of Tiny Recursive Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prediction (ARC grid transformation) | ARC train example pairs and test input grids | 2D (x, y) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Predicted test output grids (top-2 per test example) | 2D (x, y) | Capped |
| classification (halting/solved signal) | ARC task processing over grid examples (inferred) | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Halting signal indicating whether the task is solved | 0D (inferred) | Fixed (inferred) |

## Summary
The paper’s primary model-facing task is ARC grid-to-grid prediction: learning from train example pairs and producing test output grids for unseen tasks. It also describes an auxiliary classification objective via a halting head that predicts whether a task is solved. The supported modalities are dominated by 2D (x, y) grids, with an auxiliary 0D solved/not-solved signal. Dynamics are mostly capped by finite task/example structures and competition output constraints, while attention and state behavior are best characterized as static and constructed, respectively (inferred from the recursive transformer plus task-embedding design).

## Evidence
### Task: prediction (ARC grid transformation)
- "During competition submissions, this pre-trained model was fully fine-tuned on the train example pairs of the test tasks. This fine-tuned model was then used to predict test example outputs, using a majority voting method." (Section 2 Methods)
- "Entrants are allowed to submit two output grid predictions per test example, and the best output grid is the one that counts." (Section 4.1)
- Inference: `Capped` input dynamics is inferred because inputs are finite per task ("train example pairs of the test tasks"). `Static` attention is inferred from fixed, predefined augmentation-based evaluation ("TRM uses a majority voting method to make test output predictions, defaulting to a vote over 1,000 augmented versions of each task."). `Constructed` state is inferred from trainable task abstractions ("In TRM, each task and any augmented variant of that task is assigned a task id and has its own trainable embedding."). (Section 2.2; Section 2.4.2)

### Task: classification (halting/solved signal)
- "TRM includes a halting head designed to indicate whether a task has been solved (which primarily serves to stop recursions early during training)." (Section 5.3)
- "for ARC AGI II tasks, although the halting head does learn to detected solved train examples, it does not learn to effectively detect solved evaluation test examples." (Section 5.3)
- Inference: This is classified as `classification` with `0D` and `Fixed` output because the halting head emits a solved/not-solved decision signal. Input dimension/dynamics and attention/state are inferred to match the same recursive grid-processing setup used for ARC tasks (recursive transformer plus task-id embeddings). (Section 2 Methods; Section 2.4.2; Section 5.3)
