1. **Number of distinct tasks evaluated:** 4 tasks

   - "We evaluated AlphaProof on a comprehensive suite of formal mathematics benchmarks, all manually formalized in Lean, spanning advanced high-school to elite Olympiad and university-level problems. Our evaluation suite comprises: (1) a corrected version of the publicly available miniF2F benchmark [20] (high-school mathematics competitions); (2) formalimo, a benchmark of all non-geometry (because of specific Mathlib library limitations for Olympiad-style geometry, see section 6.3 in Methods) historical IMO problems internally formalized by experts; and (3) the public Putnam benchmark [21] (undergraduate Putnam Mathematical Competition problems)." (Section 2.1 "Benchmarks")
   - "To assess AlphaProof's capabilities on an unseen competition, we applied it to the problems from the 2024 IMO, operating as the core reasoning engine within a complete problem-solving pipeline." (Section 2.5 "Performance at the 2024 International Mathematical Olympiad")

2. **Number of trained model instances required to cover all tasks:** 2 models

   - "At its core is the proof network, a 3 billion parameters encoder-decoder transformer model [15, 16], that learns to interpret the observed Lean tactic state (fig. 1b) and generate two outputs: a policy, suggesting promising tactics to apply next, and a value function, estimating the expected return  $G_t$  (as defined in section 1.1)." (Section 1.2 "Prover Agent")
   - "Given specific Mathlib library limitations for Olympiad-style geometry (see Methods, IMO-style Geometry), the geometry problem (P4) was addressed using the specialized AlphaGeometry 2 system [3]." (Section 2.5 "Performance at the 2024 International Mathematical Olympiad")

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{4\ \text{tasks}}{2\ \text{models}} = 2
}
$$
