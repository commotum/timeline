# DiffSAT: Differential MaxSAT Layer for SAT Solving (2024)
Source: a34395-2024.pdf

## Core reasons
- DiffSAT builds a MaxSAT network layer that differentiates clause loss and acts as the solver during forward/backward propagation, explicitly turning SAT solving into neural computation rather than classification.
- The MaxSAT layer's backward pass picks variables from falsified clauses by their gradients (and falls back to random falsified clauses when stuck), offering a differentiable search step that updates assignments in the steepest direction.

## Evidence extracts
- "DiffSAT introduces a network layer, the MaxSAT layer, that differentiates the proposed loss function and facilitates the search for satisfying assignments." (Introduction)
- "The backward pass begins by initializing the gradient to zero for all variables (line 1). If í µí±¦í µí± is false, indicating the presence of unsatisfied clauses, we obtain the set of variables í µí°¼ that are present in the falsified clauses. Once we have obtained the candidate set í µí°¼ (line 5), we proceed to select the best variable from this set using a criterion based on its gradient. Specifically, we compute the gradient for each variable in the candidate set and choose the variable with the largest absolute gradient (line 10); if no variable satisfies the above condition, indicating that the search is stuck in a local optimum, the MaxSAT layer randomly selects a variable from a falsified clause (line 12) and artificially assigns a gradient with the same sign as the selected variable (line 13). This criterion guides the updates towards satisfying more clauses at each iteration, as selecting the variable with the largest absolute gradient pushes the loss quantity to decrease in the steepest direction." (Section 3.3 (Differential MaxSAT Layer))

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
