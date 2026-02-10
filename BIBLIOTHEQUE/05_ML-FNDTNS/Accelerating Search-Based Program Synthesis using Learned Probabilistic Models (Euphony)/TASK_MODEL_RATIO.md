1. **Number of distinct tasks evaluated: 3**

> "We chose synthesis tasks from three different application domains: i) string manipulation (STRING), ii) bit-vector manipulation (BITVEC), and iii) circuit transformation (CIRCUIT)." (Section 5.1, "Synthesis Tasks")

2. **Number of trained model instances required to cover all tasks: 4**

> "Since Euphony learns and applies a probabilistic model within a domain, we require a sufficient number of problem instances in the domain (> 200) for training and testing purposes." (Section 5.1, "Synthesis Tasks")

> "For each domain, we use all problems that the baseline tool EUSOLVER could solve within 10 minutes each as the training set, and we train the model for that domain using the solutions found by EUSOLVER." (Section 5.2, "Effectiveness of EUPHONY")

> "Both solvers use the divide-and-conquer strategy (described in Section 3.4) for this domain." (Section 5.2, "Result for BITVEC.")

> "It takes two statistical program models: the term model  $G_q^T$  and the predicate model  $G_q^P$ , and the two heuristic functions based on those grammars, respectively." (Section 3.4.2, "Divide-and-Conquer Enumeration")

> "models separately using the two grammars. Those models guide the search for terms and predicates, respectively." (Section 3.4.2, "Divide-and-Conquer Enumeration")

From these statements: STRING = 1 model, BITVEC = 2 models, CIRCUIT = 1 model; total = 4 models.

3. **Task–Model Ratio**

$$
\boxed{
\frac{3\ \text{tasks}}{4\ \text{models}} = 0.75
}
$$
