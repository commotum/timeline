1. **Number of distinct tasks evaluated:** 3

> "We test our MAXSAT layer approach in three domains that are traditionally difficult for neural networks: learning the parity function with single-bit supervision, learning  $9\times 9$  Sudoku solely from examples, and solving a \"visual Sudoku\" problem that generates the logical Sudoku solution given an input image of a Sudoku puzzle." (Section 4: Experiments)

2. **Number of trained model instances required to cover all tasks:** 3

> "Hence, for a sequence of length L, we construct our model to contain a sequence of L-1 SATNet layers with tied weights (similar to a recurrent network)." (Section 4.1: Learning parity)

> "Our model architecture consists of a single SATNet layer with 300 auxiliary variables and low rank structure m=600..." (Section 4.2: Sudoku (original and permuted))

> "Our architecture for this problem uses a convolutional neural network connected to a SATNet layer." (Section 4.3: Visual Sudoku)

A single jointly trained model instance that covers all three tasks is **Not specified in the paper.**

3. **Task–Model Ratio:**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
