1. **Number of distinct tasks evaluated:** 46

   - "Following [25], we select 46 games where DQN significantly outperforms a random agent. 41 games are used for training and 5 games are held out for out-of-distribution generalization experiments." (Section 3.3)

2. **Number of trained model instances required to cover all tasks:** 6

   - "Specifically, we investigate whether a single model – with a single set of parameters – can be trained to act in multiple environments from large amounts of expert and non-expert experience. We consider training on a suite of 41 Atari games [9, 25] for their diversity, informally asking \"Can models learn something universal from playing many video games?\"." (Section 1 Introduction)
   - "We hence devise our own evaluation setup by pretraining DT, CQL, CPC, BERT, and ACL on the full datasets of the 41 training games with 100M steps each, and fine-tuning one model per held-out game using 1% (1M steps) from each game." (Section 4.5)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{46\ \text{tasks}}{6\ \text{models}} = 7.67
}
$$
