1. Number of distinct tasks evaluated: 3

- "We train implicit networks on three benchmark image classification datasets licensed under CC-BY-SA: SVHN (Netzer et al. 2011), MNIST (LeCun, Cortes, and Burges 2010), and CIFAR-10 (Krizhevsky and Hinton 2009)." (Section: Classification)

2. Number of trained model instances required to cover all tasks: 3

- "Table 2 compares performance between using the standard Jacobian-based backpropagation and JFB. The experiments are performed on all the datasets described in Section." (Section: Comparison to Jacobian-based Backpropagation)
- "| JFB            | MNIST   | 17.6                   | 0                                  | 99.4       |" (Section: Comparison to Jacobian-based Backpropagation, Table 2)
- "|                | SVHN    | 36.9                   | 0                                  | 94.1       |" (Section: Comparison to Jacobian-based Backpropagation, Table 2)
- "|                | CIFAR10 | 146.6                  | 0                                  | 93.67      |" (Section: Comparison to Jacobian-based Backpropagation, Table 2)
- "Not specified in the paper." (for a single jointly trained model handling all datasets)

3. Task-Model Ratio

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
