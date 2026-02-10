1. **Number of distinct tasks evaluated:** 3

> "For our experiments, we focus on arithmetic, commonsense reasoning, and grade school math to demonstrate STaR's breadth." (Section 4 Experiments)

2. **Number of trained model instances required to cover all tasks:** 3

> "For arithmetic problems, we first generate a dataset of 50,000 randomly sampled questions (uniformly over the digit lengths) in the format introduced by [5]." (Section 4.1 Experimental Protocol)

> "For each of the 9,741 questions in the training set of CommonsenseQA, we add the question to the few-shot rationale prompt, and prompt the model to generate the rationale and answer for that question." (Section 4.1 Experimental Protocol)

> "Note that, in training, it was necessary to cap the number of training steps at the 30th iterations (after 7912 steps), to prevent the training process from becoming prohibitively long. The results were reached after 36 iterations for STaR without rationalization and an additional 10 iterations with rationalization." (Section 4.5 Mathematical Reasoning in Language: Grade School Math)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
