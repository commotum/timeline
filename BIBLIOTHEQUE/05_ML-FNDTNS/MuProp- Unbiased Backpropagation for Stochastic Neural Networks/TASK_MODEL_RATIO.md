1. **Number of distinct tasks evaluated:** 3

> "The two tasks are to predict the lower half of an MNIST digit given the top half, and to predict multiple facial expressions from an average face using Toronto Face dataset (TFD); the output distribution in both tasks exhibits complex multi-modality." (Section 5.1 STRUCTURED OUTPUT PREDICTION)

> "In the second set of experiments, we apply MuProp to variational training of generative models." (Section 5.2 Variational training of generative models)

2. **Number of trained model instances required to cover all tasks:** 3

> "The first task does not make use of an inference network and involves direct optimization of an approximation to the expected objective. The second task involves training a sigmoid belief network jointly with an inference network by maximizing the variational lower bound on the intractable log-likelihood." (Section 5 EXPERIMENTS)

> "For MNIST, a fixed learning rate is chosen from  $\{0.003, 0.001, ..., 0.00003\}$ , and the best test result is reported for each method. For the TFD dataset, the learning rate is chosen from the same list, but each learning rate is 10 times smaller." (Section 5.1 STRUCTURED OUTPUT PREDICTION)

A single jointly trained model instance that simultaneously covers all distinct tasks without task-specific training: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
