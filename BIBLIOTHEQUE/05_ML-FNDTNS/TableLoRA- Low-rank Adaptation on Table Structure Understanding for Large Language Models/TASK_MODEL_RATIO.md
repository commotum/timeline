1. **Number of distinct tasks evaluated:** 2

   - "We conducted experiments on three models across four datasets that encompass QA and fact
verification tasks on the tables." (§1 Introduction)
   - "The first three are Table QA datasets, where the input consists of a table and a related query, and the task is to answer the query based on the table, with the output being the answer to the question." (§4.1 Experiment Setup)
   - "The last dataset, TabFact, is for fact verification, where the input is a table and a related statement, and the task is to determine the truthfulness of the statement based on the table, with the output being the judgment result." (§4.1 Experiment Setup)

2. **Number of trained model instances required to cover all tasks:** 4

   - "In the main experiments, we evaluated TableLoRA, the original LoRA (Hu et al., 2022), and full parameter fine-tuning across three models and four datasets." (§4.2 Main Results)
   - "The LoRA fine-tuning used eight LoRA ranks with an alpha value of 16 and a dropout rate of 0.1. In all cases, training employed a batch size of 8 per device, with gradient accumulation steps of 2, a learning rate of 5e-6, and a cosine scheduler. Training was conducted for three epochs with a maximum sequence length of 4,000 tokens on the TabFact dataset and 1,000 tokens on other datasets." (§B.2 LoRA Fine-Tuning)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{2\ \text{tasks}}{4\ \text{models}} = 0.5
}
$$
