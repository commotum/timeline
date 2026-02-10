1. **Number of distinct tasks evaluated:** 3

   "We briefly introduce the three sequence prediction tasks/datasets that we employ to evaluate the DEQ approach in Section 5." (Section F: Task Descriptions)

   "Copy memory task." (Section F: Task Descriptions)

   "**Penn Treebank.** The Penn Treebank (PTB) corpus [31] is a commonly used dataset for characterand word-level language modeling." (Section F: Task Descriptions)

   "**WikiText-103.** The training corpus of WikiText-103 (WT103) [35] is about 110 times larger than PTB, with a vocabulary size over 260K." (Section F: Task Descriptions)

2. **Number of trained model instances required to cover all tasks:** 3 models

   "we evaluate DEO on both synthetic stress tests and realistic large-scale language modeling" (Section 5: Experiments)

   "Following the set of hyperparameters used by [8] for TrellisNet, we evaluate the DEQ-TrellisNet instantiation on word-level language modeling with the PTB corpus." (Section 5.2: Performance on Penn Treebank)

   "On the much larger scale WT103 corpus (about 100x larger than PTB), the DEQ-TrellisNet achieves better test perplexity than the original deep TrellisNet." (Section 5.2: Performance on WikiText-103)

   "In this task, each sequence  $\mathbf{x}_{1:(T+20)}$  is 1-dimensional and has length T+20" (Section F: Task Descriptions, Copy memory task)

   "When used for word-level language modeling, PTB contains about 888K words at training, with a vocabulary size of 10,000." (Section F: Task Descriptions, Penn Treebank)

   "The training corpus of WikiText-103 (WT103) [35] is about 110 times larger than PTB, with a vocabulary size over 260K." (Section F: Task Descriptions, WikiText-103)

   A single jointly trained model instance that handles all three tasks: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
