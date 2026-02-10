1. **Number of distinct tasks evaluated:** 4

> "In this paper, we build on advances in generative pretraining of language models to apply reward learning to four natural language tasks: continuing text with positive sentiment or physically descriptive language, and summarization tasks on the TL;DR and CNN/Daily Mail datasets." (Abstract)

> "We have demonstrated RL fine-tuning of language models to four NLP tasks: stylistic continuation with high sentiment or physically descriptive language, and summarization on the CNN/Daily Mail and TL;DR datasets." (Section 5. Conclusion)

2. **Number of trained model instances required to cover all tasks:** 4

> "We apply our method to two continuation tasks defined by human judgments:" (Section 3.1.2)

> "We also applied our method to two summarization tasks: the CNN/Daily Mail dataset of Hermann et al. (2015) and the TL;DR dataset of Völske et al. (2017)." (Section 3.2)

> "For stylistic continuation tasks we perform supervised fine-tuning of the language model to the BookCorpus dataset of Zhu et al. (2015) prior to RL fine-tuning; we train from scratch on WebText, supervised fine-tune on BookCorpus, then RL fine-tune to our final task." (Section 2.1. Pretraining details)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
