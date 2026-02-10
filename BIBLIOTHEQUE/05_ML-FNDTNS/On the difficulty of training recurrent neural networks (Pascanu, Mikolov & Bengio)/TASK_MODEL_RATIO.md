1. **Number of distinct tasks evaluated:** 9

   "We consider the temporal order problem as the prototypical pathological problem, extending our results to the other proposed tasks afterwards." (Section 4.1.1)

   "SGD-CR is able to solve (100% success on the lengths listed below, for all but one task) other pathological problems from Hochreiter and Schmidhuber (1997), namely the addition problem, the multiplication problem, the 3-bit temporal order problem, the random permutation problem and the noiseless memorization problem in two variants (when the pattern needed to be memorized is 5 bits in length and when it contains over 20 bits of information; see Martens and Sutskever (2011))." (Section 4.1.2)

   "We address the task of polyphonic music prediction, using the datasets Piano-midi.de, Nottingham and MuseData described in Boulanger-Lewandowski et al. (2012) and language modelling at the character level on the Penn Treebank dataset (Mikolov et al., 2012). We also explore a modified version of the task, where we require to predict the 5th character in the future (instead of the next)." (Section 4.2)

2. **Number of trained model instances required to cover all tasks:** 9 models

   "For every task we used 5 different runs (with different random seeds)." (Section 4.1.2)

   "For the first 4 problems we used a single model for lengths up to 200, while for the noiseless memorization we used a different model for each sequence length (50, 100, 150 and 200)." (Section 4.1.2)

   "Furthermore, we can train a single model to deal with any sequence of length 50 up to 200 (by providing sequences of different random lengths in this interval for different MSGD steps)." (Section 4.1.1)

   A single jointly trained model instance covering all listed tasks: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{9\ \text{tasks}}{9\ \text{models}} = 1
}
$$
