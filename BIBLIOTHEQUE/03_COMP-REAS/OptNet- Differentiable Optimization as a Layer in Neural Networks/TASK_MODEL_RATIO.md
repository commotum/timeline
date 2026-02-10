1. **Number of distinct tasks evaluated:** 3
- "our goal in this case is to denoise a noisy 1D signal given training data consistency of noisy and clean signals generated from the same distribution." (Section 4.2. Total variation denoising)
- "In this section we consider the integration of QP OptNet layers into a traditional fully connected network for the MNIST problem." (Supplementary Material, Section A. MNIST Experiment)
- "Finally, we present the main illustrative example of the representational power of our approach, the task of learning the game of Sudoku." (Section 4.4. Sudoku)

2. **Number of trained model instances required to cover all tasks:** 3
- Denoising model instance: "An alternative approach to denoising is by learning from data. A function  $f_{\theta}(x)$  parameterized by  $\theta$  can be used to predict the original signal." (Section 4.2.2. BASELINE: LEARNING WITH A FULLY-CONNECTED NEURAL NETWORK)
- MNIST model instance: "Specifically we use a FC600-FC10-SoftMax fully connected network and compare it to a FC600-FC10-Optnet10-SoftMax network, where the numbers after each layer indicate the layer size." (Supplementary Material, Section A. MNIST Experiment)
- Sudoku model instance: "We trained these models using ADAM (Kingma & Ba, 2014) to minimize the MSE (which we refer to as \"loss\") on a dataset we created consisting of 9000 training puz-\n\nzles, and we then tested the models on 1000 different heldout puzzles." (Section 4.4. Sudoku)
- Single jointly trained model spanning denoising, MNIST, and Sudoku: Not specified in the paper.

3. **Task–Model Ratio**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
