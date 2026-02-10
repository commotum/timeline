1. **Number of distinct tasks evaluated:** 3

   "Intuitively, encoding positional information explicitly is crucial for enabling transformer language models to predict the next token in a sequence." (Section 3: Experiment Setup)

   "Specifically, we use the tokens' last hidden representation after each transformer layer, produced by the evaluated LM, and train a 2-layer feed-forward ReLU network to predict the absolute position (0 to 1023) of each token (i.e., as a multiclass classification problem)." (Section 5: Analysis, "NoPos models acquire positional information")

   "We tested this corollary by training a masked language model based on RoBERTa large (Liu et al., 2019) on the Pile (see App. C for hyperparameters)." (Section 6: Conjecture)

2. **Number of trained model instances required to cover all tasks:** 3

   "To test this intuition, we compared the validation set perplexity of models trained from scratch with no explicit positional information (denoted as *NoPos*) to those trained with the various positional encoding methods discussed in Section 2." (Section 3: Experiment Setup)

   "Specifically, we use the tokens' last hidden representation after each transformer layer, produced by the evaluated LM, and train a 2-layer feed-forward ReLU network to predict the absolute position (0 to 1023) of each token (i.e., as a multiclass classification problem)." (Section 5: Analysis, "NoPos models acquire positional information")

   "We tested this corollary by training a masked language model based on RoBERTa large (Liu et al., 2019) on the Pile (see App. C for hyperparameters)." (Section 6: Conjecture)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
