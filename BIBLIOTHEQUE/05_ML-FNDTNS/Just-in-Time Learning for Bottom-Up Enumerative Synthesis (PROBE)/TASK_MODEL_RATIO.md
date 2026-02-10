1. **Number of distinct tasks evaluated:** 3
   - "We evaluate Probe on three different application domains: string manipulation (String), bit-vector manipulation (BitVec), and circuit transformations (Circuit)." (Section 6.1 Experimental Setup)

2. **Number of trained model instances required to cover all tasks:** 3
   - "Probe takes as input an inductive SyGuS problem  $\mathcal{G}$ ,  $\mathcal{E}$ . It starts by initializing the PCFG with CFG  $\mathcal{G}$  and a uniform distribution  $p_u$ , which assigns every production rule  $R = N \to \beta$  the probability  $p(R) = 1/|\mathcal{R}(N)|$ ." (Section 5.1 Algorithm summary)
   - "At the start of each CEGIS iteration, we initialize an independent instance of Probe starting from a uniform grammar." (Section 6.1 Experimental Setup)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
