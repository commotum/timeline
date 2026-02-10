1. **Number of distinct tasks evaluated**

"We perform experiments with layer normalization on 6 tasks, with a focus on recurrent neural networks: image-sentence ranking, question-answering, contextual language modelling, generative modelling, handwriting sequence generation and MNIST classification." (Section 6, "Experimental results")

**Answer:** 6

2. **Number of trained model instances required to cover all tasks**

Evidence that task-specific models are used across the six task sections:

"In this experiment, we apply layer normalization to the recently proposed order-embeddings model..." (Section 6.1, "Order embeddings of images and language")

"...we train an unidirectional attentive reader model..." (Section 6.2, "Teaching machines to read and comprehend")

"...we train two models on the BookCorpus dataset..." (Section 6.3, "Skip-thought vectors")

"We evaluate the effect of layer normalization on a DRAW model..." (Section 6.4, "Modeling binarized MNIST using DRAW")

"...we performed handwriting generation tasks..." (Section 6.5, "Handwriting sequence generation")

"...we investigated layer normalization in feed-forward networks... permutation invariant MNIST classification problem." (Section 6.6, "Permutation invariant MNIST")

A single jointly trained model that performs all six tasks is not described. **Not specified in the paper.**

**Answer:** 6

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{6\ \text{tasks}}{6\ \text{models}} = 1
}
$$
