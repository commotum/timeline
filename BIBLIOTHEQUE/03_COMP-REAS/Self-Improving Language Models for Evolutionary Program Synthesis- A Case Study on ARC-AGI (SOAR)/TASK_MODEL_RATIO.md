1. **Number of distinct tasks evaluated:** 2 tasks

   - “...examining three key questions: (1) how to learn better program sampling, (2) how to learn better program refinement, and (3) whether and how these capabilities can be learned jointly.” (Section 4.2, *Learning to sample and refine programs*)
   - “Should we train separate models for sampling and refinement, or can a single model learn both effectively?” (Section 4.2, *Positive synergy between sample and refine tasks*)

2. **Number of trained model instances required to cover all tasks:** 1 model

   - “At each iteration i, we alternate between: (1) Sample & Refine search phase: Using model  $\theta_i$  to sample and refine programs...” (Section 3.4, *Closing the loop: iterative self-improvement on training and testing tasks*)
   - “Table 4 shows that joint finetuning outperforms both base models and task-specific finetuning—for both sampling and search performance.” (Section 4.2, *Positive synergy between sample and refine tasks*)
   - “| fine-both    | fine-both    | 39.79     | 44.42                |” (Table 4, Section 4.2)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{2\ \text{tasks}}{1\ \text{model}} = 2
}
$$
