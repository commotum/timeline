1. **Number of distinct tasks evaluated:** 6

   "The objective is to classify the digit images into their correct digit class." (Section **A.1 Details for dropout training**)

   "Fig. 2: The frame *classification* error rate on the core test set of the TIMIT benchmark." (Main text, Fig. 2)

   "To get the frame *recognition* rate, the class probabilities that the neural network outputs for each frame are given to a decoder which knows about transition probabilities between HMM states and runs the Viterbi algorithm to infer the single best sequence of HMM states." (Main text)

   "CIFAR-10 is a benchmark task for object recognition." (Main text)

   "ImageNet is an extremely challenging object recognition dataset consisting of thousands of high-resolution images of thousands of classes of object (11)." (Main text)

   "The Reuters dataset contains documents that have been labeled with a hierarchy of classes." (Main text)

2. **Number of trained model instances required to cover all tasks:** 6

   "The pretrained RBMs were used to initialize the weights in a neural network. The network was then finetuned with dropout-backpropagation." (Section **B.2 Dropout Finetuning**, TIMIT frame classification model)

   "To get the frame *recognition* rate, the class probabilities that the neural network outputs for each frame are given to a decoder which knows about transition probabilities between HMM states and runs the Viterbi algorithm to infer the single best sequence of HMM states." (Main text, TIMIT recognition decoder)

   "We show results for 4 nets (784-800-800-10, 784-1200-1200-10, 784-2000-2000-10, 784-1200-1200-1200-10)." (Section **A.1 Details for dropout training**, MNIST)

   "We trained a neural network using dropout-backpropagation and compared it with standard backpropagation. We used a 2000-2000-1000-50 architecture." (Section **C Experiments on Reuters**)

   "Our model for CIFAR-10 with dropout is similar, but because dropout imposes a strong regularization on the network, we are able to use more parameters." (Section **G Models for CIFAR-10**)

   "Our model for ImageNet with dropout is a CNN which is trained on  $224 \times 224$  patches randomly extracted from the  $256 \times 256$  images, as well as their horizontal reflections." (Section **H Models for ImageNet**)

3. **Task–Model Ratio**

$$
\boxed{
\frac{6\ \text{tasks}}{6\ \text{models}} = 1
}
$$
