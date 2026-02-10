1. **Number of distinct tasks evaluated:** 1
"We evaluate the hybrid solver neuro-minisat (described in §4) and the original MiniSat solver minisat on the 400 problems from the main track of SATCOMP-2018, with the same 5,000 second timeout used in the competition." (Section: `5 Solver Experiments`)
"Glucose. As a follow-up experiment and sanity-check, we made the same modifications to Glucose 4.1 and evaluated in the same way on SATCOMP-2018." (Section: `5 Solver Experiments`)
"Z3. Lastly, we made the same modifications to Z3, except we once again altered the NeuroCore schedule, this time from exponential backoff in terms of user-time to geometric backoff in terms of the number of conflicts." (Section: `5 Solver Experiments`)

2. **Number of trained model instances required to cover all tasks:** 1
"Thus, fine-tuning the network is relatively unimportant, and we only ever trained with a single set of hyperparameters." (Section: `3 Neural Network Architecture` - `Training Neuro Core`)
"We approximate this regime by evaluating the same trained network discussed above on the set of 303 (non-public) hard scheduling problems that were included in the data generation process along with SATCOMP 2013-2017." (Section: `5 Solver Experiments` - `A more favorable regime`)

3. **Task-Model Ratio**

$$
\boxed{
\frac{1\ \text{task}}{1\ \text{model}} = 1
}
$$
