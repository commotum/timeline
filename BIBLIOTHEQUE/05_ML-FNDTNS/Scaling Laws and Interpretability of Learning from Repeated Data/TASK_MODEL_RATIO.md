1. **Number of distinct tasks evaluated:** 4

Evidence: "A small, held-back portion of the original dataset (yellow in left figure), not including any repeated data, is used as a test set and is the test loss reported in all subsequent figures." (Figure 1, page 1); "We constructed a simple copying eval, the loss on the first paragraph of Harry Potter copied 11 times." (Section 1.1 Summary of Results); "we evaluated the models on their prefix matching score, repeated sequences of random tokens and observed the degree to which attention heads attend to earlier tokens that are preceded by a token that matches the present token." (Section 1.1 Summary of Results); "Repeated text data causes a small but still disproportionate performance drop out of distribution, as measured by cross entropy loss on Python code." (Section 1.1 Summary of Results)

2. **Number of trained model instances required to cover all tasks:** 2

Evidence: "To systematically study repeated data, we trained transformer [Vaswani et al., 2017] language models on mostly unique data plus a small fraction of repeated data (Figure 1), varying the repeated dataset size, model size, and fraction of tokens trained on repeated data over 2-3 orders of magnitude. All models were trained for 100B tokens." (Section 1.1 Summary of Results); "Code models were trained or fine-tuned on 45B tokens of Python for 2.2 epochs." (Section 3 Methods)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{4\ \text{tasks}}{2\ \text{models}} = 2
}
$$
