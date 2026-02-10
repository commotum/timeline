# TOWARDS PRINCIPLED UNSUPERVISED LEARNING (Not specified in the paper.)
Source: Towards Principled Unsupervised Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unsupervised permutation recovery (MNIST permutation task) | Permuted MNIST digit images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | MNIST digit images in original pixel arrangement | 2D (x, y) (inferred) | Fixed (inferred) |
| Unsupervised cipher decipherment (character/word symbol correspondence) | Scrambled text symbol streams (characters or words) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Recovered symbol correspondence / decoded symbol stream | 1D (t) (inferred) | Fixed (inferred) |
| Digit classification (MNIST) with ODM/GAN training | MNIST digit images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Digit class labels (10-way) | 0D (inferred) | Fixed (inferred) |
| One-shot domain adaptation by test-time inference | 1-MNIST test digit images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Inferred MNIST-domain images x* for downstream classification | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper covers unsupervised task domains in both image and text settings: 2D digit-image mappings and 1D token-stream symbol correspondence. It evaluates image-to-image alignment (MNIST permutation recovery and one-shot domain adaptation), token-symbol correspondence recovery in ciphers, and image-to-label classification on MNIST. From the described datasets and model interfaces, input/output dynamics are Fixed and attention is Static across tasks (inferred). State is mostly Direct, with one-shot adaptation using per-test optimization over latent variables and parameters, so its state is Constructed (inferred).

## Evidence
### Task: Unsupervised permutation recovery (MNIST permutation task)
- "We chose  $\mathcal{Y}$  to be the distribution of MNIST digits, and  $\mathcal{X}$  to be the distribution of MNIST digits whose pixels are permuted (with the same permutation on all digits)." (Section 4.1)
- "We call it the *MNIST permutation task*." (Section 4.1)
- "The goal of the task is to learn the unknown permutation using no supervised data." (Section 4.1)
- "We found that the dual autoencoder was easily able to recover the permutation without using any input-output examples" (Section 4.1)
- Inference: Input/output dimensions and dynamics are inferred as 2D (x, y), Fixed because the task is defined on MNIST digit images and implemented with a fixed-size 784-100-100-100-784 architecture; attention/state are inferred as Static/Direct because the described dual autoencoder is a feed-forward mapping without runtime input selection or persistent external memory. (Section 4.1)

### Task: Unsupervised cipher decipherment (character/word symbol correspondence)
- "We also tested the dual autoencoder on a character and a word cipher" (Section 4.1.1)
- "The goal is to find the hidden correspondence between the symbols in both streams." (Section 4.1.1)
- "the input (as well as the the desired output) is represented with a bag of 10 consecutive characters from a random sentence from the text corpus." (Section 4.1.1)
- Inference: Input/output are inferred as 1D (t) token streams and Fixed dynamics from the fixed-size bag-of-10 representation and fixed architectures; attention/state are inferred as Static/Direct because no runtime observation selection or persistent constructed memory is specified. (Section 4.1.1)

### Task: Digit classification (MNIST) with ODM/GAN training
- "Next, we evaluated the GAN-based model on an artificial OCR task that was constructed from the MNIST dataset." (Section 4.2)
- "The goal of this problem was to train an MLP F that maps MNIST digits to 10-dimensional vectors without using any input-output examples." (Section 4.2)
- "We trained an MLP F to map each MNIST digit into a 10-dimensional vector representing their classification." (Section 4.2)
- Inference: The output dimension is inferred as 0D because the task intent is classification (digit class decision), despite implementation as a 10-dimensional vector; image dimensions/dynamics are inferred as 2D (x, y), Fixed from MNIST and the fixed 784-300-300-10 classifier; attention/state are inferred as Static/Direct from the feed-forward classifier/discriminator setup. (Section 4.2)

### Task: One-shot domain adaptation by test-time inference
- "Suppose that the training distribution is  $\mathcal{D}$ , but we are provided with a single test sample  $y \sim \mathcal{D}'$  where  $\mathcal{D}'$  is an unknown test distribution." (Section 5)
- "If y contains enough information to uniquely identify a simple function G that maps samples  $y \sim \mathcal{D}'$  to samples  $x \sim \mathcal{D}$ , we will be able to classify samples  $y \sim \mathcal{D}'$  without any extra training" (Section 5)
- "we can try to infer the unknown x by solving the following optimization problem:  $x^* = \operatorname{argmax}_{x,\theta} P_{\theta}(y|x) P(x)$" (Section 5)
- "We emphasize that this optimization is run *from scratch* for each new test case y." (Section 5)
- Inference: Input/output are inferred as 2D (x, y), Fixed from MNIST/1-MNIST image setup; attention is inferred as Static because the method optimizes over full image likelihood terms rather than runtime input selection; state is inferred as Constructed because each test case constructs latent variables/parameters (x and \(\theta\)) through per-sample optimization before downstream classification. (Section 5)
