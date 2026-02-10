1. Number of distinct tasks evaluated: 8

> "For binary classification, five benchmark datasets are used: Musk1 and Musk2 [23] datasets for molecular activity predictions; Fox, Elephant, and Tiger datasets for image classification. For multi-class classification two datasets are used: multiple-instance MNIST (MIL-MNIST) [13] dataset for handwritten digit classification; MIL-based CIFAR-10 datasets [15] for object recognition. Additionally, the experiments are also conducted for real-world Colon Cancer detection histopathology dataset [24]." (Section 1, Introduction)

2. Number of trained model instances required to cover all tasks: 8

> "The experiments on benchmark datasets are performed using five runs of 10-fold cross-validation, and average performance is reported. For the MIL-based MNIST dataset, the experiments are performed with 1000 test bags and different numbers of training bags (50, 100, 150, 200, 300, and 400). The experiments are repeated 50 times for each train and test set, and average results are compared with existing state-of-the-art techniques. Similarly, the experiments are repeated thirty times with different training and testing data for MIL-based CIFAR-10 datasets, and average performance is reported. On the Colon Cancer dataset, we performed a 5-fold cross-validation, and average results are presented." (Section 4.1.5, Evaluation measure)

Single jointly trained model instance covering all tasks: Not specified in the paper.

3. Task–Model Ratio

$$
\boxed{
\frac{8\ \text{tasks}}{8\ \text{models}} = 1
}
$$
