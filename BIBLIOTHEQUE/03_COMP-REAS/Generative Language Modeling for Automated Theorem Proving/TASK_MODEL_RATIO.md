1. **Number of distinct tasks evaluated:** 4

> "We report the performance  $\\operatorname{Perf}_{a,e,d}^{valid}(\\theta)$  of a model  $\\theta$ , as the percentage of proofs found by this procedure within the *valid* or *test* set."
(Section 4.4 Evaluation)

> "We describe below the synthetic datasets we designed and report in section 5 the sample complexity associated with these synthetic tasks."
(Section 4.6 Synthetic Datasets)

> "| Model                                  | 9-digit addition | 9-digit division | Ring equalities |"
(Section 5.5 Sample Complexity, Table 11)

2. **Number of trained model instances required to cover all tasks:** 1

> "We explain the improvement over *MetaGen-IL* (despite not relying on forward proving data generation techniques) by our use of a simpler architecture (one unique Transformer vs 3 separate GRU networks); a more straightforward objective (direct auto-regressive generation of the full tactic as text vs separate premise selection and generation of the substitutions); more learnable parameters (160m vs 300k (3 2-layers bi-directional GRUs with 128 hiddens)); and more compute at training as well as test time."
(Section 5.1 Baselines)

> "Not having to introduce a separate value head greatly simplifies the overall architecture."
(Section 4.7 Learned Value Function)

> "At each iteration we entirely re-train the model on both objectives at the same time on the dataset constructed as follows:"
(Section 4.7.1 Iterative Data Generation and Training)

> "| 700m <i>policy+value</i> (iteration 2) | 92% (100)        | 47% (100)        | 88% (100)       |"
(Section 5.5 Sample Complexity, Table 11)

> "Table 12: Performance of our 700m model *policy+value* (iteration 2) as we double the number of attempts a per proposition (with d=256)."
(Section 5.6 Results)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{4\ \text{tasks}}{1\ \text{model}} = 4
}
$$
