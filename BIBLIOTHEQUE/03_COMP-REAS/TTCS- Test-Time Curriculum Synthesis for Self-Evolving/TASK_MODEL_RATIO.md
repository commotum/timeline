1. **Number of distinct tasks evaluated:** 9

- "We apply **TTCS** to each benchmark individually and then evaluate to demonstrate its effectiveness." (Section 5.1 Experimental Setting)
- "(1) **Competition-Level Mathematical Benchmarks:** We employ the AMC23, AIME24, and AIME25 as a rigorous testbed for advanced reasoning ability [20]." (Section 5.1 Experimental Setting)
- "(2) **Fundamental Mathematical Benchmarks:** Complementarily, we include MATH-500 [12], Minerva [19], and OlympiadBench [10] to assess fundamental mathematical proficiency across diverse problem types (See Appendix A.1)." (Section 5.1 Experimental Setting)
- "To investigate the cross-domain generalization of TTCS, we conducted evaluations on challenging general-domain reasoning benchmarks, including MMLU-Pro [38] and SuperGPQA [34]." (Section 5.3 Analysis)
- "Additional results, including BBEH [18], are provided in Appendix B.1." (Section 5.3 Analysis)

2. **Number of trained model instances required to cover all tasks:** 18

- "We apply **TTCS** to each benchmark individually and then evaluate to demonstrate its effectiveness." (Section 5.1 Experimental Setting)
- "TTCS consists of two agents: a Synthesizer policy  $\pi_{\phi}$  and a Solver policy  $\pi_{\theta}$ , both initialized from the same pretrained model." (Section 4 Test-Time Curriculum Synthesis)
- "Without Synthesizer Training. We freeze the synthesizer and use a static pretrained model to generate questions while allowing only the solver to evolve." (Section 5.4 Ablation Study)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{9\ \text{tasks}}{18\ \text{models}} = 0.5
}
$$
