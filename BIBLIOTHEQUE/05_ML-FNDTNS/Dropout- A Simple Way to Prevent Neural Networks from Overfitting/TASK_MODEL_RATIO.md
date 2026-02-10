1. **Number of distinct tasks evaluated: 8**
"We used five image data sets to evaluate dropout—MNIST, SVHN, CIFAR-10, CIFAR-100 and ImageNet." (Section 6.1)
"Next, we applied dropout to a speech recognition task. We use the TIMIT data set which consists of recordings from 680 speakers covering 8 major dialects of American English reading ten phonetically-rich sentences in a controlled noise-free environment." (Section 6.2)
"The task is to take a bag of words representation of a document and classify it into 50 disjoint topics." (Section 6.3)
"Given the RNA features, the task is to predict the probability of three splicing related events that biologists care about." (Section 6.4)

2. **Number of trained model instances required to cover all tasks: 8**
"We trained dropout neural networks for classification problems on data sets in different domains." (Section 6)
"For this data set, we applied dropout to convolutional neural networks (LeCun et al., 1989)." (Section 6.1.2)
"Dropout neural networks were trained on windows of 21 log-filter bank frames to predict the label of the central frame." (Section 6.2)
"To test the usefulness of dropout in the text domain, we used dropout networks to train a document classifier." (Section 6.3)
"A two layer dropout network with 1024 units in each layer was trained on this data set." (Appendix B.6)
"Not specified in the paper." (Single jointly trained model covering all tasks)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{8\ \text{tasks}}{8\ \text{models}} = 1
}
$$
