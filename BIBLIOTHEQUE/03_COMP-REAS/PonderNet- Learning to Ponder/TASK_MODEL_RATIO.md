1. **Number of distinct tasks evaluated:** 22

   - "In this section we are reporting results on the parity task as introduced in the original ACT paper (Graves, 2016)." (Section 3.1 Parity)
   - "We then turn our attention to the bAbI question answering dataset (Weston et al., 2015), which consists of 20 different tasks." (Section 3.2 **bAbI**)
   - "Finally, we tested PonderNet on the Paired associative inference task (PAI) (Banino et al., 2020)." (Section 3.3 Paired associative inference)

2. **Number of trained model instances required to cover all tasks:** 3

   - "For this experiment we used the Parity task as explained by Graves (2016)." and "All the models used the same architecture, a simple RNN with a single hidden layer containing 128 tanh units and a single logistic sigmoid output unit." (Appendix B.1 Training and evaluation details)
   - "In particular we trained our model on the joint 10k training set." and "For this experiment we used the English Question Answer dataset Weston et al. (2015)." (Section 3.2 **bAbI**; Appendix C.1 Training and evaluation details)
   - "For this task we used the dataset published in Banino et al. (2020), also the task is available at https://github.com/deepmind/deepmind-research/tree/master/memo" (Appendix D.1 PAI - Task details)
   - A single jointly trained model instance spanning Parity + bAbI + PAI is Not specified in the paper.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{22\ \text{tasks}}{3\ \text{models}} = \frac{22}{3}
}
$$
