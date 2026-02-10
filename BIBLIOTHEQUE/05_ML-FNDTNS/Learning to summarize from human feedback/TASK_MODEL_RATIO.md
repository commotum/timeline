1. **Number of distinct tasks evaluated:** 1 task

   - "Our goal in this paper is to advance methods for training language models on objectives that more closely capture the behavior we care about. To make short-term progress towards this goal, we focus on abstractive English text summarization, as it has a long history in the NLP community [16, 8, 54, 59, 50], and is a subjective task where we believe it is difficult to quantify summary quality without human judgments." (Section 1: Introduction)
   - "**Task.** We define our ground-truth task as producing a model that generates summaries fewer than 48 tokens long that are as good as possible, according to our judgments." (Section 3.2: Datasets and task)

2. **Number of trained model instances required to cover all tasks:** 1 model

   - "Instead of training on CNN/DM, we study the transfer performance of our human feedback models to CNN/DM after being trained to summarize Reddit posts." (Section 3.2: Datasets and task)
   - "Our human feedback models can also generate excellent summaries of CNN/DM news articles without any further training (Figure 4)." (Section 4.2: Transfer to summarizing news articles)

3. **Task–Model Ratio = (1) / (2):** 1

$$
\boxed{
\frac{1\ \text{task}}{1\ \text{model}} = 1
}
$$
