# Dropout: A Simple Way to Prevent Neural Networks from Overfitting (Not specified in the paper.)
Source: Dropout- A Simple Way to Prevent Neural Networks from Overfitting.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | images | 2D (x, y) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | class labels | 0D | Fixed (inferred) |
| classification | speech frames (log-filter bank frame windows) | 1D (t) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | phone label (central frame) | 0D | Fixed (inferred) |
| classification | bag-of-words documents | 1D (t) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | topic label | 0D | Fixed (inferred) |
| prediction | RNA features | 1D (t) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | probability distribution over splicing states (tissue types) | 1D (t) | Fixed (inferred) |
| reconstruction | images (MNIST) | 2D (x, y) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | reconstructed images | 2D (x, y) | Fixed (inferred) |

## Summary
The paper applies dropout across multiple supervised classification domains: 2D image classification, 1D speech frame (phone) classification, and 1D text document classification, plus a biological prediction task that outputs probability distributions over splicing states. It also analyzes reconstruction in an MNIST autoencoder, indicating coverage of image reconstruction tasks. The inputs are fixed-size grids or vectors (2D images; 1D feature/bag-of-words windows), producing fixed-size labels or probability vectors, with attention and state dynamics inferred as static and constructed based on fixed inputs and multi-layer networks.

## Evidence
### Task: classification
- "The MNIST data set consists of  $28 \times 28$  pixel handwritten digit images." (Section 6.1.1 MNIST)
- "The task is to classify the images into 10 digit classes." (Section 6.1.1 MNIST)
- "The CIFAR-10 and CIFAR-100 data sets consist of 32 × 32 color images drawn from 10 and 100 categories respectively." (Section 6.1.3 CIFAR-10 AND CIFAR-100)
- Inference: In/Out Dynamics marked Fixed and Attention Static because inputs/classes are fixed-size (e.g., "$28 \times 28$" images and "10 digit classes"); State Constructed because models have "multiple non-linear hidden layers." (Sections 6.1.1 MNIST; 1. Introduction)

### Task: classification
- "Next, we applied dropout to a speech recognition task." (Section 6.2 Results on TIMIT)
- "Dropout neural networks were trained on windows of 21 log-filter bank frames to predict the label of the central frame." (Section 6.2 Results on TIMIT)
- Inference: In/Out Dynamics marked Fixed and Attention Static because the input is a fixed window ("windows of 21 log-filter bank frames") and output is a single label; State Constructed because models have "multiple non-linear hidden layers." (Section 6.2 Results on TIMIT; 1. Introduction)

### Task: classification
- "we used dropout networks to train a document classifier." (Section 6.3 Results on a Text Data Set)
- "The task is to take a bag of words representation of a document and classify it into 50 disjoint topics." (Section 6.3 Results on a Text Data Set)
- "a vocabulary of 2000 words comprising of 50 categories" (Appendix B.5 Reuters)
- Inference: In/Out Dynamics marked Fixed and Attention Static because the input is a fixed-size bag-of-words vector ("vocabulary of 2000 words") and output is a fixed set of topics; State Constructed because models have "multiple non-linear hidden layers." (Appendix B.5 Reuters; 1. Introduction)

### Task: prediction
- "The task is to predict the occurrence of alternative splicing based on RNA features." (Section 6.4 Comparison with Bayesian Neural Networks)
- "For each input, the target consists of 4 softmax units (one for tissue type)." (Appendix B.6 Alternative Splicing)
- "Each softmax unit has 3 states (*inc*, *exc*, *nc*)" (Appendix B.6 Alternative Splicing)
- Inference: In/Out Dynamics marked Fixed and Attention Static because inputs/outputs are fixed-size feature and state sets ("1014 RNA features" and "4 softmax units"); State Constructed because models have "multiple non-linear hidden layers." (Appendix B.6 Alternative Splicing; 1. Introduction)

### Task: reconstruction
- "Figure 7a shows features learned by an autoencoder on MNIST with a single hidden layer of 256 rectified linear units without dropout." (Section 7.1 Effect on Features)
- "Both autoencoders had similar test reconstruction errors." (Section 7.1 Effect on Features)
- Inference: In/Out Dynamics marked Fixed and Attention Static because the autoencoder uses fixed-size MNIST images ("$28 \times 28$" images); State Constructed because models have "multiple non-linear hidden layers." (Sections 7.1 Effect on Features; 6.1.1 MNIST; 1. Introduction)
