1. **Number of distinct tasks evaluated:** 4

   "We show empirical evidence of our claims regarding ordering, and on the modifications to the seq2seq framework on benchmark language modeling and parsing tasks, as well as two artificial tasks – sorting numbers and estimating the joint probability of unknown graphical models." (ABSTRACT)

   "In our initial attempt to solve (9), we considered a simplified version of the language modeling task described in Section 5.1.1." (Section 5.2.1)

2. **Number of trained model instances required to cover all tasks:** 4 models

   "For each ordering we trained a different model." (Section 5.1.1)

   "We thus tried to train a small model using depth first traversal (which matches the baseline of Vinyals et al. (2015b)) and another one using breadth first traversal" (Section 5.1.2)

   "For each problem, we trained two LSTMs for 10,000 mini-batch iterations to model the joint probability" (Section 5.1.4)

   "In order to verify if our model handles sets more efficiently than the vanilla seq2seq approach, we ran the following experiment on artificial data for the task of sorting numbers" (Section 4.4)

   Single jointly trained model covering all tasks: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
