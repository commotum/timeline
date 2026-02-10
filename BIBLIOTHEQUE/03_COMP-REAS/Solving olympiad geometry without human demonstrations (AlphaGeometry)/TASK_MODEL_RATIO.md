1. **Number of distinct tasks evaluated:** `1`
- "We focus on Euclidean plane geometry and exclude topics such as geometric inequalities and combinatorial geometry." (Main text, before section `Synthetic theorems and proofs generation`)
- "Among all non-combinatorial geometry-related problems, 75% can be represented, resulting in a test set of 30 classical geometry problems." (Section: `An olympiad-level benchmark for geometry`)
- "On a larger and more diverse test set of 231 geometry problems, which covers textbook exercises, regional olympiads and famous theorems, we find that baselines in Table 1 remain at the same performance rankings, with AlphaGeometry solving almost all problems (98.7%), whereas Wu's method solved 75% and DD + AR + human-designed heuristics solved 92.2%, as reported in Extended Data Fig. 6b." (Section: `Proving results on IMO-AG-30`)

2. **Number of trained model instances required to cover all tasks:** `1`
- "AlphaGeometry is a neuro-symbolic system that uses a neural language model, trained from scratch on our large-scale synthetic data, to guide a symbolic deduction engine through infinite branching points in challenging problems." (Main text abstract)
- "We first pretrained the language model on all 100 million synthetically generated proofs, including ones of pure symbolic deduction. We then fine-tuned the language model on the subset of proofs that requires auxiliary constructions, accounting for roughly 9% of the total pretraining data, that is, 9 million proofs, to better focus on its assigned task during proof search." (Section: `Language model pretraining and fine-tuning`)
- "For each problem, we used a pool of four GPU workers, each hosting a copy of the transformer language model to divide the work between alternative beams, and a pool of 10,000 CPU workers to host the symbolic solvers, shared across all beams across all 30 problems." (Methods: `Language model architecture and training`)

3. **Task–Model Ratio**

$$
\boxed{
\frac{1\ \text{task}}{1\ \text{model}} = 1
}
$$
