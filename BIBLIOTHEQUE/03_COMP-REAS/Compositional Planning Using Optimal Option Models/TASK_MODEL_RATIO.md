1. **Number of distinct tasks evaluated:** 2

   "We illustrate our framework for compositional planning using two hierarchical MDPs: the Tower of Hanoi problem, and the  $Nine\ Rooms$  problem." (Section 6, **Empirical Results**)

   "We also use stochastic variants in which each action causes the intended move with probability 1-p, or with probability p randomly selects another legal move (Tower of Hanoi, p = 0.4), or remains in the current state (Nine Rooms, p = 0.05)." (Section 6, **Empirical Results**)

2. **Number of trained model instances required to cover all tasks:** 2

   "For the Tower of Hanoi, we use m = 3N + 1 subgoal value models." (Section 6, **Empirical Results**)

   "For the Nine Rooms, we use 12(n-1) subgoal value models." (Section 6, **Empirical Results**)

   "At each iteration k, the algorithm updates a set of m option models  $\mathcal{M}^k = \{\mathbf{M}_1^k,...,\mathbf{M}_m^k\}$ , containing one option model for every subgoal." (Section 5, **Option-Option Model Iteration**)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
