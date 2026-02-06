# ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION (Not specified in the paper.)
Source: Adam- A Method for Stochastic Optimization.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| optimization of stochastic objective functions | stochastic objective function with parameters theta; gradients g_t | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | parameter vector theta | Not specified in the paper. | Not specified in the paper. |
| classification | 784 dimension image vectors (MNIST) | 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | class label | 0D (inferred) | Fixed (inferred) |
| classification (inferred) | bag-of-words (BoW) feature vectors of IMDB movie reviews (10,000 dimension) | 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | class label (inferred) | 0D (inferred) | Fixed (inferred) |
| autoencoding (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper's primary task domain is stochastic optimization of objective functions via gradient-based parameter updates, and it evaluates classification on fixed-length MNIST image vectors and fixed-length IMDB bag-of-words review vectors. A variational autoencoder is also trained as an additional use case, but the input/output modality for that task is not specified. Across the explicitly described classification tasks, inputs are 1D fixed vectors and outputs are 0D class labels (inferred for IMDB), while attention dynamics are not specified; the optimizer itself maintains constructed internal state via moving averages (inferred).

## Evidence
### Task: optimization of stochastic objective functions
- "algorithm for first-order gradient-based optimization of stochastic objective functions" (Abstract)
- "Require: f(theta): Stochastic objective function with parameters theta" (Algorithm 1)
- "return theta_t (Resulting parameters)" (Algorithm 1)
- Inference: State Dynamic marked Constructed because "The algorithm updates exponential moving averages of the gradient (m_t) and the squared gradient (v_t)" (Section 2)

### Task: classification
- "L2-regularized multi-class logistic regression using the MNIST dataset." (Section 6.1)
- "The logistic regression classifies the class label directly on the 784 dimension image vectors." (Section 6.1)
- Inference: Input labeled 1D Fixed and output labeled 0D Fixed because the paper specifies "784 dimension image vectors" and a "class label." (Section 6.1)

### Task: classification (inferred)
- "We examine the sparse feature problem using IMDB movie review dataset" (Section 6.1)
- "bag-of-words (BoW) feature vectors including the first 10,000 most frequent words." (Section 6.1)
- "Logistic regression training negative log likelihood on MNIST images and IMDB movie reviews with 10,000 bag-of-words (BoW) feature vectors." (Figure 1)
- Inference: Task/output labeled classification with 0D labels because logistic regression is used on IMDB reviews; input labeled 1D Fixed due to the fixed 10,000-dimension BoW vectors. (Section 6.1; Figure 1)

### Task: autoencoding (inferred)
- "learning a Variational Auto-Encoder (VAE)" (Figure 4)
- "training a variational autoencoder (VAE)" (Section 6.4)
- Inference: Task labeled autoencoding based on the paper's explicit use of a Variational Auto-Encoder; input/output modality is not specified. (Figure 4; Section 6.4)
