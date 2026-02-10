1. **Number of distinct tasks evaluated:** 2

- "4. Experimental Validation: We provide theoretical complexity analysis demonstrating subquadratic scaling and present experimental results on hierarchical classification and long-range dependency tasks showing competitive performance with dramatically reduced parameters (section VI)." (Section I. INTRODUCTION)
- "#### 1. Hierarchical Sequence Classification" (Section VI. EXPERIMENTS, A. Experimental Setup)
- "#### 2. Long-Range Dependency Task" (Section VI. EXPERIMENTS, A. Experimental Setup)

2. **Number of trained model instances required to cover all tasks:** 2 models

- "where  $f_{out}$  is a task-specific output function (e.g., softmax for classification) and  $\mathbf{W}_{out} \in \mathbb{R}^{d_{out} \times d_h}$  projects to output dimension." (Section III. RESONANT SPARSE GEOMETRY NETWORKS, G. Resonance and Output)
- "We generate sequences of length L=64 with feature dimension d=32, divided into C=20 classes." (Section VI. EXPERIMENTS, A.1. Hierarchical Sequence Classification)
- "This task has 10 classes (10% random baseline)." (Section VI. EXPERIMENTS, A.2. Long-Range Dependency Task)
- "For all experiments, RSGN uses N=256 nodes, hidden dimension  $d_h=128$ , embedding dimension d=3, and K=5 propagation steps (7 for long-range task)." (Section VI. EXPERIMENTS, A.4. RSGN Configuration)
- Explicit statement that one jointly trained model handles both tasks: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
