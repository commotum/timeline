1. **Number of distinct tasks evaluated:** 6

> “Our first goal is to learn a classifier that approximates  $\phi$ . Given a distribution  $\Psi$  over SAT problems, we can construct datasets  $\mathcal{D}_{\text{train}}$  and  $\mathcal{D}_{\text{test}}$  with examples of the form  $(P,\phi(P))$  by sampling problems  $P\sim\Psi$  and computing  $\phi(P)$  using an existing SAT solver. At test time, we get only the problem P and the goal is to predict  $\phi(P)$ , i.e. to determine if P is satisfiable. Ultimately we care about the solving task, which also includes finding solutions to satisfiable problems.” (Section 2 PROBLEM SETUP, “Classification task”)

> “Moreover, NeuroSAT generalizes to novel distributions; after training only on random SAT problems, at test time it can solve SAT problems encoding graph coloring, clique detection, dominating set, and vertex cover problems, all on a range of distributions over small random graphs.” (ABSTRACT)

> “However, when we train the same architecture on a dataset in which each unsatisfiable problem has a small subset of clauses that are already unsatisfiable (called an *unsat core*), it learns to detect these unsat cores instead of searching for satisfying assignments.” (Section “FINDING UNSAT CORES”)

2. **Number of trained model instances required to cover all tasks:** 2

> “Note: for the entire rest of the paper, NeuroSAT refers to the specific trained model that has only been trained on  $\mathbf{SR}(\mathbf{U}(10,40))$ .” (Section 5 PREDICTING SATISFIABILITY)

> “The same neural network architecture can also be used to help construct proofs for unsatisfiable problems. When we train it on a different dataset in which every unsatisfiable problem contains a small contradiction (call this trained model *NeuroUNSAT*), it learns to detect these contradictions instead of searching for satisfying assignments.” (Section 1 Introduction)

> “NeuroSAT (trained on SR(U(10,40))) can find satisfying assignments but is not helpful in constructing proofs of unsatisfiability.” (Section “FINDING UNSAT CORES”)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{6\ \text{tasks}}{2\ \text{models}} = 3
}
$$
