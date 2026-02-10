1. **Number of distinct tasks evaluated: 2**
- "To verify the effects of internal covariate shift on training, and the ability of Batch Normalization to combat it, we considered the problem of predicting the digit class on the MNIST dataset (LeCun et al., 1998a)." (Section 4.1 "Activations over time")
- "We applied Batch Normalization to a new variant of the Inception network (Szegedy et al., 2014), trained on the ImageNet classification task (Russakovsky et al., 2014)." (Section 4.2 "ImageNet classification")

2. **Number of trained model instances required to cover all tasks: 2**
- "We used a very simple network, with a 28x28 binary image as input, and 3 fully-connected hidden layers with 100 activations each." and "The last hidden layer is followed by a fully-connected layer with 10 activations (one per class) and cross-entropy loss." (Section 4.1 "Activations over time")
- "The network has a large number of convolutional and pooling layers, with a softmax layer to predict the image class, out of 1000 possibilities." (Section 4.2 "ImageNet classification")

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
