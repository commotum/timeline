1. **Number of distinct tasks evaluated:** 5

> "Experimental results are provided for four synthetic problems: determining the parity of binary vectors, applying binary logic operations, adding integers, and sorting real numbers." (Abstract)
>
> "We also present character-level language modelling results on the Hutter prize Wikipedia dataset." (Abstract)
>
> "We tested recurrent neural networks (RNNs) with and without ACT on four synthetic tasks and one real-world language processing task." (Section 3 Experiments)

2. **Number of trained model instances required to cover all tasks:** 5

> "A logarithmic grid search over time penalties was performed for each experiment, with 20 randomly initialised networks trained for each value of  $\tau$ ." (Section 3 Experiments)
>
> "The network architecture was a simple RNN with a single hidden layer containing  $128 \ tanh$  units and a single sigmoidal output unit, trained with binary cross-entropy loss on minibatches of size 128." (Section 3.1 Parity)
>
> "The network architecture was single-layer LSTM with 128 cells. The output was a single sigmoidal unit, trained with binary cross-entropy, and the minibatch size was 16." (Section 3.2 Logic)
>
> "The network was single-layer LSTM with 512 memory cells." (Section 3.3 Addition)
>
> "The network was single-layer LSTM with 512 cells. The output layer was a size 15 softmax," (Section 3.4 Sort)
>
> "LSTM networks were used with a single layer of 1500 cells and a size 256 softmax classification layer." (Section 3.5 Wikipedia Character Prediction)

3. **Task-Model Ratio**

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$
