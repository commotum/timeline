1. **Number of distinct tasks evaluated: 4**

> "Following Wang et al. (2025), we test our approach on the following datasets: Sudoku-Extreme (Wang et al., 2025), Maze-Hard (Wang et al., 2025), ARC-AGI-1 (Chollet, 2019) and, ARC-AGI-2 (Chollet et al., 2025)." (Section 5, Results)

> "While our approach led to better generalization on 4 benchmarks..." (Section 6, Conclusion)

2. **Number of trained model instances required to cover all tasks: 4 models**

> "For Sudoku-Extreme and Maze-Hard, we train for 60k epochs with learning rate 1e-4 and weight decay 1.0. For ARC-AGI, we train for 100K epochs with learning rate 1e-4 (with 1e-2 learning rate for the embeddings) and weight decay 0.1." (Hyper-parameters and setup)

> "From the results, we see that TRM without selfattention obtains the best generalization on Sudoku-Extreme (87.4% test accuracy). Meanwhile, TRM with self-attention generalizes better on the other tasks (probably due to inductive biases and the overcapacity of the MLP on large 30x30 grids). TRM with self-attention obtains 85.3% accuracy on Maze-Hard, 44.6% accuracy on ARC-AGI-1, and 7.8% accuracy on ARC-AGI-2 with 7M parameters." (Section 5, Results)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
