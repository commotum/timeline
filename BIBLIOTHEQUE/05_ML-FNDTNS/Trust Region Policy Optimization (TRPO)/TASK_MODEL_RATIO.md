1. **Number of distinct tasks evaluated:** 11

   - "The following models are included in our evaluation:" (Section 8.1)
   - "- 1. Swimmer. 10-dimensional state space, linear reward for forward progress and a quadratic penalty on joint effort to produce the reward  $r(x,u) = v_x 10^{-5} \|u\|^2$ ." (Section 8.1)
   - "- Hopper. 12-dimensional state space, same reward as the swimmer, with a bonus of +1 for being in a nonterminal state." (Section 8.1)
   - "- 3. Walker. 18-dimensional state space." (Section 8.1)
   - "To establish a standard baseline, we also included the classic cart-pole balancing problem..." (Section 8.1)
   - "We tested our algorithms on the same seven games reported on in (Mnih et al., 2013) and (Guo et al., 2014), which are" (Section 8.2)

2. **Number of trained model instances required to cover all tasks:** 11 models

   - "Our algorithms (bottom rows) were run once on each task, with the same architecture and parameters." (Table 1, Section 8.2)
   - A single jointly trained model that covers all 11 tasks: "Not specified in the paper."

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{11\ \text{tasks}}{11\ \text{models}} = 1
}
$$
