1. **Number of distinct tasks evaluated:** 5

- "#### 4.1 DUAL AUTOENCODERS ON MNIST PERMUTATION TASK" (Section 4.1)
- "We also tested the dual autoencoder on a character and a word cipher, tasks that were also considered by Knight et al. (2006)." (Section 4.1.1)
- "#### 4.2 GENERATIVE ADVERSARIAL NETWORKS FOR MNIST CLASSIFICATION" (Section 4.2)
- "#### 5 ONE-SHOT LEARNING AND DOMAIN ADAPTATION" (Section 5)

2. **Number of trained model instances required to cover all tasks:** 5

- "We used a dual autoencoder whose architecture is 784-100-100-100-784" (Section 4.1)
- "For the character-level experiments, we used the architecture 100-25-25-100, and for the word-level experiments we used 1000-100-1000" (Section 4.1.1)
- "We trained an MLP F to map each MNIST digit into a 10-dimensional vector representing their classification." (Section 4.2)
- "Our concrete model choices are the following: P(x) is implemented with a next-row-prediction LSTM with three hidden layers that has been trained to fit the MNIST distribution with the binary cross entropy loss, and P(y|x) is a small convolutional neural network (CNN) with one hidden layer" (Section 5)

3. **Task–Model Ratio**

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$
