# Resonant Sparse Geometry Networks (2026)
Source: Resonant Sparse Geometry Networks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hierarchical sequence classification | real-valued feature sequences | 1D (t) (inferred) | Fixed | Dynamic (inferred) | Constructed (inferred) | class label | 0D (inferred) | Fixed |
| Long-range dependency classification | real-valued feature sequences | 1D (t) (inferred) | Fixed | Dynamic (inferred) | Constructed (inferred) | class label | 0D (inferred) | Fixed |

## Summary
RSGN is evaluated on two sequence classification tasks: hierarchical sequence classification and long-range dependency classification. Both tasks use 1D temporal sequences with fixed input lengths in the benchmark definitions (L=64 and L=128) and fixed class outputs (20-way and 10-way), yielding 0D class-label outputs. The model behavior described in the paper supports Dynamic attention (input-dependent routing/ignition) and Constructed state (iterative node-state propagation) across these tasks.

## Evidence
### Task: Hierarchical sequence classification
- "#### 1. Hierarchical Sequence Classification" (Section VI.A.1)
- "The task involves classifying sequences based on patterns organized at three hierarchical levels:" (Section VI.A.1)
- "We generate sequences of length L=64 with feature dimension d=32, divided into C=20 classes." (Section VI.A.1)
- "In contrast, RSGN adapts its active computation graph for each input through the ignition mechanism, with different inputs potentially activating entirely different subsets of nodes." (Section II.A)
- "Each node maintains state variables that evolve on different timescales, separating fast activation dynamics from slow structural plasticity." (Section III.A)
- Inference: `1D (t)` is inferred from sequence-indexed inputs ("input sequence X = [x_1, ..., x_T]"). `Dynamic` attention is inferred from input-dependent routing/ignition. `Constructed` state is inferred from evolving node activation state variables across propagation steps. `0D` output is inferred from class-label prediction over 20 classes. (Sections III.D, VI.A.1)

### Task: Long-range dependency classification
- "#### 2. Long-Range Dependency Task" (Section VI.A.2)
- "To evaluate RSGN's ability to capture dependencies across long sequences, we designed a task where class labels depend on patterns at both the *beginning* and *end* of sequences." (Section VI.A.2)
- "Specifically, for sequences of length L=128:" (Section VI.A.2)
- "This task has 10 classes (10% random baseline)." (Section VI.A.2)
- "In contrast, RSGN adapts its active computation graph for each input through the ignition mechanism, with different inputs potentially activating entirely different subsets of nodes." (Section II.A)
- "The activation state evolves through K steps (K = 5 in experiments) according to:" (Section III.E)
- Inference: `1D (t)` is inferred from dependence on sequence positions (beginning/end). `Dynamic` attention is inferred from input-dependent routing. `Constructed` state is inferred from iterative activation-state updates used to integrate distant sequence evidence. `0D` output is inferred from 10-way class-label prediction. (Sections II.A, III.E, VI.A.2)
