1. Number of distinct tasks evaluated: 2. Quote: "We evaluate our proposed method on several benchmark datasets, using GPT-2 variants for next-token prediction and BERT models for long-context retrieval." (Section 1 Introduction)
2. Number of trained model instances required to cover all tasks: 2. Quotes: "For all next-token prediction tasks, we use the GPT-2 variants (Brown et al., 2020)." (Section 4.2 Settings) "To adapt BERT models for this task, we fine-tune them on MS-MARCO (Nguyen et al., 2016) using mined hard negatives (Xuan et al., 2020), with 1.25M samples, a batch size of 128, and a 5% learning rate warmup over one epoch, leveraging the sentence-transformers framework (Reimers and Gurevych, 2019)." (Section 5.5 Bidirectional Models)
3. Task–Model Ratio:

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
