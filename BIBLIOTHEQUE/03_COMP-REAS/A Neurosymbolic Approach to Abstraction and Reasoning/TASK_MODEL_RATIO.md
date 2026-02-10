1. **Number of distinct tasks evaluated:** 2

- ARC: "In Section 3 and Section 4, we develop two approaches and sets of results on ARC." (Section 1 Introduction)
- 24 Game: "We present preliminary results on ARC problems with this approach, as well as application to a simpler domain of solving tasks from the \"24 Game\" family of puzzles." (Section 1 Introduction)
- Section 4.4 confirms both evaluations: "We evaluate the bidirectional algorithm on a set of 18 ARC symmetry tasks" and "we evaluate the agent in a simpler domain: solving \"24 Game\" problems." (Section 4.4 Results)

2. **Number of trained model instances required to cover all tasks:** 2

- ARC-trained instance: "We trained on a set of randomly generated programs evaluated on random input grids from the ARC training set, and fine-tuned with REINFORCE" (Section 4.4 Results)
- 24-Game-trained instance: "For the supervised pretraining of our model, we train for 10,000 epochs on a dataset of randomly generated programs between depth one and four" (Section 4.4 Results)
- A single jointly trained ARC+24 model instance is **Not specified in the paper.**

3. **Task–Model Ratio**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
