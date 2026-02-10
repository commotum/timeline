1. **Number of distinct tasks evaluated:** 6

   "The two core tasks on which we study the performance of Neural GPUs are long binary addition and long binary multiplication." (Section 3.1)

   "In addition to the two main tasks above, we tested Neural GPUs on the following simpler algorithmic tasks." (Section 3.2)

   "Copying sequences is the simple task of producing on output the same sequence as on input." (Section 3.2)

   "Reversing sequences is the task of reversing a sequence of bits, n is the length of the sequence." (Section 3.2)

   "Duplicating sequences is the task of duplicating the input bit sequence on output twice, as in the example below." (Section 3.2)

   "Counting by sorting bits is the task of sorting the input bit sequence on output." (Section 3.2)

2. **Number of trained model instances required to cover all tasks:** 6

   "The same architecture as used above was able to solve all of the tasks described below..." (Section 3.2)

   "Each result we report is obtained by running a grid search over  $3^6 = 729$  instances." (Section 3.3)

   A single jointly trained model that performs all listed tasks is Not specified in the paper.

3. **Task–Model Ratio:**

$$
\boxed{
\frac{6\ \text{tasks}}{6\ \text{models}} = 1
}
$$
