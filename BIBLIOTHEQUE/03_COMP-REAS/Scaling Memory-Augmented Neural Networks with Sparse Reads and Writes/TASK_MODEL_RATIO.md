1. **Number of distinct tasks evaluated:** 24

   - "We trained each model on three of the original NTM tasks [7]. **1. Copy**: copy a random input sequence of length 1–20, **2. Associative Recall**: given 3-6 random (key, value) pairs, and subsequently a cue key, return the associated value. **3. Priority Sort**: Given 20 random keys and priority values, return" and "the top 16 keys in descending order of priority." (Section 4.2)
   - "The task was encoded using straightforward 1-hot word encodings for both the input and output. We trained a single model on all of the tasks" and Supplementary Table 1 lists tasks "1: 1 supporting fact" through "20: agent's motivations" (Supplementary G, Table 1)
   - "Finally, we demonstrate that the model is capable of learning in a non-synthetic dataset. Omniglot [12] is a dataset of 1623 characters taken from 50 different alphabets, with 20 examples of each character. This dataset is used to test rapid, or *one-shot* learning" (Section 4.5)

2. **Number of trained model instances required to cover all tasks:** 5

   - 3 models for Copy, Associative Recall, and Priority Sort (Section 4.2).
   - "We trained a single model on all of the tasks" for the 20 bAbI tasks (Supplementary G).
   - "After training all MANNs for the same length of time, a validation task with 500 characters was used to select the best run, and this was then tested on a test set" for Omniglot (Section 4.5).

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{24\ \text{tasks}}{5\ \text{models}} = 4.8
}
$$
