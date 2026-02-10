1. **Number of distinct tasks evaluated:** 1

Citations (verbatim):
"Figure 1 shows the evolution of pass@2 and pass@1000 accuracy on the 120 ARC AGI II public eval tasks during the training run, reaching a final pass@2 score of 10%, which is higher than the originally reported score of 7.8% in the TRM paper, presumably associated with stochastic behaviour during training." (Section 3.1 Pre-Training)

"Table 1: Semi-private evaluation accuracy achieved after post-training ARC AGI II checkpoints within the competition environment." (Section 3.2.2 Post-training of ARC AGI II pre-trained models during competition submissions)

2. **Number of trained model instances required to cover all tasks:** 1

Citations (verbatim):
"A recursive transformer model was pre-trained on ARC AGI II training tasks in close accordance to the Tiny Recursive Model paper [3]. During competition submissions, this pre-trained model was fully fine-tuned on the train example pairs of the test tasks. This fine-tuned model was then used to predict test example outputs, using a majority voting method." (Section 2 Methods)

"If post training requires fewer optimizer steps than training from scratch – to reach the same performance – then clearly there is meaningful inter-task learning OR at least there are shared concepts involved in training TRM." (Section 5.4 On the benefit of joint training on multiple tasks for compute efficiency)

"The effect can also be understood by pretraining on a single task versus 8 versus 120 tasks on a model of the same size. While it appears possible to achieve similar performance when training on a task individually, or combined with other tasks, it is more efficient - again for a model of fixed size - to jointly train on multiple tasks [7], showing that decomposing a post-training run into separate batches of posttraining and inference does not materially improve performance but does increase total training time." (Section 5.4 On the benefit of joint training on multiple tasks for compute efficiency)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{1\ \text{tasks}}{1\ \text{models}} = 1
}
$$
