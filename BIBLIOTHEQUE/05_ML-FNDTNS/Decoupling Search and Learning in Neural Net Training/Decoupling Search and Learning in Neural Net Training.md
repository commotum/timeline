# DECOUPLING SEARCH AND LEARNING IN NEURAL NET TRAINING

Akshay Vegesna\* Q Labs Samip Dahal\* Q Labs

# **ABSTRACT**

Gradient descent typically converges to a single minimum of the training loss without mechanisms to explore alternative minima that may generalize better. Searching for diverse minima directly in high-dimensional parameter space is generally intractable. To address this, we propose a framework that performs training in two distinct phases: search in a tractable representation space (the space of intermediate activations) to find diverse representational solutions, and gradientbased learning in parameter space by regressing to those searched representations. Through evolutionary search, we discover representational solutions whose fitness and diversity scale with compute—larger populations and more generations produce better and more varied solutions. These representations prove to be learnable: networks trained by regressing to searched representations approach SGD's performance on MNIST, CIFAR-10, and CIFAR-100. Performance improves with search compute up to saturation. The resulting models differ qualitatively from networks trained with gradient descent, following different representational trajectories during training. This work demonstrates how future training algorithms could overcome gradient descent's exploratory limitations by decoupling search in representation space from efficient gradient-based learning in parameter space.

# 1 Introduction

Neural network training is fundamentally a search process over parameter configurations, seeking those that minimize training loss while generalizing well to unseen data. The ideal approach for generalization would be exhaustive search—systematically exploring the parameter space to discover many diverse minima, then selecting those with the best generalization properties. However, the parameter space of modern neural networks is vast, containing millions or billions of dimensions. Exhaustive search of this space is computationally intractable; even with substantial compute, we could only explore a vanishingly small fraction of possible configurations. Gradient descent emerged as a practical alternative to this intractable search problem. Rather than exploring broadly, it efficiently descends from an initial point by following local gradients, finding good solutions without needing to examine the entire parameter space. This efficiency comes with a fundamental limitation: gradient descent converges to a single local minimum and cannot explore alternative regions of the loss landscape to discover the diverse solutions that may exist elsewhere. This trade-off between exhaustive search (diverse but intractable) and gradient descent (efficient but limited) motivates the need for training algorithms that can combine both approaches.

We argue that exploring diverse solutions may be fundamental to achieving better generalization. To understand why, consider what a learning algorithm that yields optimal generalization would do. Our core hypothesis (detailed here) is that given infinite compute, such an algorithm would enumerate all models consistent with the data and use a generalization prior to select the best one. Solomonoff induction (Solomonoff, 1964) is an example of this ideal—enumerating all computable models and weighting them by simplicity. While computationally intractable, this framework reveals what's missing from gradient-based training: exploration of diverse solutions through search.

Though complete synthesis of search and gradient descent extends beyond a single paper, we take a first step by decomposing the problem: instead of searching directly in parameter space, we perform explicit search in representation space—the activations of intermediate layers. This space is

<sup>\*</sup>Equal contribution. Correspondence to research@qlabs.sh

far smaller, making search tractable via evolutionary algorithms. Moreover, these searched representations prove to be learnable: neural networks can be trained through gradient descent to predict these representations. This means we can find effective intermediate representations through search, then train network parameters to produce those patterns—effectively using search to guide where gradient descent should go. By scaling search to discover diverse representational solutions and steering gradient descent toward them, future algorithms could overcome gradient descent's inability to explore.

This paper takes three steps:

- 1. **Conceptual framework** (Section 2): We recast neural network training as two decoupled phases. First, we perform evolutionary search in the representation space to discover representational solutions. Second, we use gradient-based learning in parameter space to train the network to produce these representations. This separation preserves the benefits of search while avoiding the computational intractability of searching in parameter space.
- 2. Search in representation space (Section 3): We implement evolutionary search over representations to minimize training loss and study how it scales with compute. We find that both solution fitness and diversity increase with more compute—more generations and larger populations yield better and more varied representational solutions.
- 3. **Learning in parameter space** (Section 4): We train network parameters to match the searched representations and show this method approaches the performance of SGD on MNIST, CIFAR-10, and CIFAR-100—all without backpropagating cross-entropy gradients through the network body. We analyze how performance scales with search compute and demonstrate that the resulting models are qualitatively different from those produced by SGD.

#### 2 Conceptual Framework

The most direct way to instantiate our hypothesis—searching over many models and selecting good ones based on the generalization prior—would be to search directly in parameter space. However, the choice of search space must satisfy specific principles for both search and learning.

# 2.1 Core principles for search and learning

For effective search, we argue that the space must satisfy two key principles:

- 1. Random search tractability. The space must be small enough that naive random sampling or perturbations can make progress towards high-fitness solutions.
- 2. Amortization. We want to train a sampler that learns from random perturbations and their fitness values, generalizing to produce new high-fitness solutions. We can then search around these solutions with further perturbations and amortize again, creating an iterative improvement cycle.

When a space doesn't satisfy these search principles, we fall back to gradient-based optimization when applicable.

#### 2.2 Why parameter space fails for search

Parameter space fails both search requirements. First, it is far too large for random sampling to be effective—the probability of randomly finding good parameters is vanishingly small. Second, learning to directly sample parameters of a large neural net is difficult, although this has been tried at a small scale (Wang et al., 2024).

Given these limitations, parameter space only supports learning through gradient descent, not search.

## 2.3 DECOUPLING

Good models must produce good representations. This observation suggests we can decompose the problem: instead of searching for parameters directly, we can search for good representations, then learn parameters that produce those representations using gradient descent. This breaks the intractable problem of searching in parameter space into two tractable subproblems.

Representation space satisfies both search principles. First, the space of activations is far smaller than parameter space, making random perturbations effective at finding high-fitness solutions. Second, neural networks already define predictive distributions over their own representations, providing a built-in mechanism for amortization.

Once we have discovered high-fitness representations through search, learning parameters to induce them is straightforward with gradient descent.

## 3 SEARCH

#### 3.1 How we search in Representation space

**Core algorithm.** We perform evolutionary search directly on neural network representations by searching over the layerwise activations—the actual intermediate tensors produced at key layers. Rather than optimizing parameters, we treat the activation tensors at selected layers as the primary search space, evolving these representations to minimize classification loss. The key insight is that by evolving representations at intermediate layers in a forward pass, we can discover high-quality solutions without backpropagation.

Our approach consists of two main stages: (1) initialize a population by forward propagating inputs with noise, and (2) sequentially evolve representations at each selected layer while fixing earlier ones. We evolve activations for the first convolutional block output, fix them, then evolve the next block's activations that build on these fixed representations, and so on through the network. The optimized activations found through search then serve as regression targets to train the network parameters.

**Network architecture.** We apply our method to a standard convolutional network for CIFAR-10 classification. The network consists of three convolutional blocks followed by a linear classification head. Each block contains two  $3\times 3$  convolutional layers with batch normalization and GELU activations, with  $2\times 2$  max pooling between blocks to progressively reduce spatial dimensions. The final convolutional features are flattened and fed to a linear layer that outputs class logits. The three blocks produce feature maps of dimensions  $256\times 15\times 15, 256\times 7\times 7,$  and  $256\times 3\times 3$  respectively, giving us four representation levels to optimize (three convolutional outputs and final logits). We perform search using a network initialized with Dirac initialization for convolutional layers and Kaiming Uniform initialization for other layers.

**Problem setup.** Given input batch (x,y) and a network, we perform evolutionary search at L=4 specific points in the architecture: the outputs of the three convolutional blocks  $(\ell \in \{0,1,2\})$  and the final logits  $(\ell=3)$ . We refer to these architectural points as search layers since each convolutional block contains multiple internal layers but we only perform search at the block outputs. For the remainder of this section and the next, layer refers to these search layers unless otherwise specified. Let  $H^{(\ell)}$  denote the activations at layer  $\ell$ . The population at layer  $\ell$  is  $\mathcal{P}_{\ell} = \{H_i^{(\ell)}\}_{i=1}^{n_{\text{pop}}}$ , representing  $n_{\text{pop}}$  candidate activations. We use negative cross-entropy as our fitness function (equivalently, maximizing log-likelihood), evaluating how well each candidate's final predictions match the true labels. All selection, mutation, and fitness evaluation operations are performed independently for each image in the batch.

**Population initialization.** We initialize the population at layer 0 by sampling  $n_{\rm pop}$  noisy variants of the inputs to the first layer. The noise is channel-wise and scaled to input statistics, yielding  $n_{\rm pop}$  distinct  $H^{(0)}$  samples that serve as the starting point for evolution.

**Layer-wise forward evolution.** We evolve layers sequentially from  $\ell=0$  to L-1. When evolving layer  $\ell$ , we fix the representations at all earlier layers  $H^{(0)},\ldots,H^{(\ell-1)}$  to their previously evolved values, evolve only  $H^{(\ell)}$ , then forward propagate through the remaining network layers to recompute representations at later layers. This ensures that each layer's search builds upon the optimized

solutions found for previous layers, creating a compositional optimization process where later layers benefit from earlier improvements. Figure 1 illustrates this layer-wise evolution process, showing how we progressively evolve populations at each layer while fixing earlier layer representations (shown in red).

**Evolution mechanics.** For each generation at layer  $\ell$ , we select the top-k candidates from the population independently for each image in the batch based on fitness. This per-image selection (rather than batch-wide) accelerates convergence by maintaining diversity across different data points. From these parents, we: (1) retain all k parents unchanged, (2) create  $C_{\rm exp}$  exploratory samples with high mutation strength ( $\alpha \times$  exploration boost) to discover new high-fitness regions and avoid diversity collapse, and (3) generate  $C_{\rm ref}$  refinement samples with standard mutation strength ( $\alpha$ ) for local improvement. The next generation consists of parents  $\cup$  exploratory samples  $\cup$  refinement samples.

**Reproduction operators.** The exploratory and refinement samples are created using the same three-step process: genetic modification at layer  $\ell$ , forward propagation, and fitness evaluation. The only difference is the mutation strength  $\alpha$ .

For genetic modification, we first apply crossover  $\tilde{H}^{(\ell)} = \frac{1}{2}(H_{p_1}^{(\ell)} + H_{p_2}^{(\ell)})$ , where  $p_1, p_2$  are randomly selected parents. We then add Gaussian mutation  $\epsilon \sim \mathcal{N}(0, (\alpha \sigma_{\text{parent}})^2)$ , where  $\sigma_{\text{parent}}$  is the per-channel standard deviation of the parent activations. For convolutional layers, we use channel-wise noise (preserving spatial coherence), apply spatial smoothing via repeated  $3\times 3$  average pooling (to reduce high-frequency artifacts), and normalize to zero mean and unit variance. We found spatial smoothing to be crucial for learnability of these representations with gradient descent, and normalization for convergence of evolution. For logits, we simply add element-wise noise and recenter.

After modifying layer  $\ell$ , we forward propagate through the network's blocks to compute  $H^{(\ell+1)}, \ldots, H^{(L-1)}$ , then evaluate fitness using the final classification loss. This ensures every mutation is evaluated in the context of the full network.

## Pseudocode (Python-style)

```
# Inputs: x, y, pop_size, top_k, c_exp, c_ref, gens_per_layer
# Initialize population with channel-wise noise
population = []
for _ in range(pop_size):
   h0 = x + channel_noise(x)
   population.append(propagate_through_network(h0))
fitness = evaluate_fitness(population, y)
# Evolve each layer sequentially
for layer in range(num layers):
    for gen in range(gens_per_layer[layer]):
        # Select top-k parents per image
        parents = select_top_k_per_image(population, fitness, k=top_k)
        # Exploration: high mutation strength (alpha * boost)
        explorers = reproduce(parents, n=c exp, mutation="high")
        # Refinement: standard mutation strength (alpha)
        refiners = reproduce(parents + explorers, n=c_ref, mutation="standard")
        # New population with downstream layers recomputed
        population = parents + explorers + refiners
        population = recompute from layer(population, start=layer+1)
        fitness = evaluate_fitness(population, y)
    # Keep best individual for next layer
    population = keep_best(population)
```

![](_page_4_Figure_0.jpeg)

Figure 1: Layerwise forward evolution for  $H^{(0)}$ ,  $H^{(1)}$ , and  $H^{(2)}$ . At each layer  $\ell$ , we evolve a population  $\mathcal{P}_{\ell}$  (shown within dashed boxes), keep the best as  $H^{(\ell)}$  (red), and fix it for subsequent layers. After mutation, fitness is evaluated by completing the remaining forward pass. Representation sizes:  $H^{(0)} \in \mathbb{R}^{256 \times 15 \times 15}$ ,  $H^{(1)} \in \mathbb{R}^{256 \times 7 \times 7}$ ,  $H^{(2)} \in \mathbb{R}^{256 \times 3 \times 3}$ .

# 3.2 SCALING WITH COMPUTE

**Fitness scaling.** We study how search quality improves with compute by varying either the population size or the number of generations while holding the other fixed. For each convolutional block representation  $\ell \in \{0,1,2\}$  we evolve only  $H^{(\ell)}$ , then recompute  $H^{(\ell+1)},\ldots,H^{(L-1)}$  before scoring fitness (we report cross-entropy; lower is better). For each configuration we average the best individual's loss per image across 1000 random data points in the training set of CIFAR-100.

Figure 2 shows that loss decreases with more compute along both axes—population size and number of generations. As expected, the later layer (Block 2) optimizes more easily than the earlier ones (Blocks 0 and 1).

**Diversity scaling.** We also study how the diversity of solutions produced by evolution scales with compute. For this, we perform multiple independent evolutionary search runs and collect the top candidate from each run, provided its predicted probability for the correct class exceeds 0.5. Following prior work (Skean et al., 2025), we then compute the effective number of distinct solutions,  $N_{\rm eff}$ , derived from the collision entropy of the cosine similarity Gram matrix:

$$H_{\text{coll}} = -\log \frac{\sum_{i,j} K_{ij}^2}{\left(\sum_i K_{ii}\right)^2}, \qquad N_{\text{eff}} = \exp(H_{\text{coll}}),$$

where K is the pairwise cosine similarity matrix of candidate representations. Intuitively,  $N_{\rm eff}$  is the effective number of completely orthogonal solutions; in practice, there usually are many more partially overlapping solutions, so this quantity is a conservative estimate of diversity.

![](_page_5_Figure_0.jpeg)

Figure 2: Fitness scales with compute on CIFAR-100. Mean cross-entropy vs. population size (left) and vs. generations (right).

![](_page_5_Figure_2.jpeg)

Figure 3: Diversity grows with compute across convolutional block representations  $\ell \in \{0,1,2\}$  on CIFAR-100. Effective number of solutions  $N_{\rm eff}$  as a function of the number of independent evolutionary runs.

Figure 3 shows that  $N_{\rm eff}$  steadily increases as we aggregate results from more independent evolutionary search runs. Early convolutional block representations show the strongest growth in  $N_{\rm eff}$ , consistent with their larger representational capacity, while later blocks exhibit slower growth.

This demonstrates that search naturally produces both optimization and diversity scaling with compute.

# 4 LEARNING

#### 4.1 Why representations alone are not enough

Searching representations by itself is not sufficient. Two further requirements must hold: (i) the searched representations must be learnable by the network's layers, and (ii) learning those representations must lead to generalization on unseen data. In other words, searched representations should provide a path to models that generalize. Our hypothesis is that regressing to searched representations yields such models, and regressing to different representational solutions yields different model solutions.

#### 4.2 Learning from Searched Representations

**Caching.** We run evolutionary search once over the training set and cache the results. The search is performed using an untrained initialized network. For each training example (x, y), we store the best solution from search: the full sequence of representations  $\{\hat{H}^{(\ell)}\}_{\ell=0}^{L-1}$ , with the searched output probability distribution defined from the searched logits by  $\hat{p}(\cdot \mid x) = \operatorname{softmax}(\hat{H}^{(L-1)}(x))$ . These cached representations become fixed regression targets—we never re-run search during training.

**Objective.** We design our training objective to ensure the convolutional body learns exclusively from the searched representations, not from classification gradients. We minimize the following with gradient descent:

$$\mathcal{L}(\theta,\phi) = \frac{1}{L-1} \sum_{\ell=0}^{L-2} \frac{1}{B} \sum_{b=1}^{B} \|H_{\theta}^{(\ell)}(x_b) - \hat{H}^{(\ell)}(x_b)\|_{2}^{2} + \frac{\lambda}{B} \sum_{b=1}^{B} \text{KL} \Big( \hat{p}(\cdot \mid x_b) \|p_{\phi} \Big( \cdot \mid \text{sg}(H_{\theta}^{(L-2)}(x_b)) \Big) \Big)$$

Here  $\theta$  parameterizes the convolutional layers,  $\phi$  parameterizes the classification head, and  $sg(\cdot)$  denotes the stop-gradient operator. This operator prevents KL gradients from flowing into the convolutional layers—only the head parameters  $\phi$  receive gradients from the KL term. The convolutional representations are thus shaped entirely by the MSE regression targets, avoiding collapse to standard backpropagation. This layer-wise MSE objective bears similarity to target propagation methods (Lee et al., 2015), although our targets come from different sources.

Capacity of the blocks. Target maps of intermediate layers produced by search are very high dimensional and involve large representational jumps. We empirically find that increased network capacity is helpful to learn these searched representations. While search uses blocks with 2 convolutional layers each, during learning we expand each block to contain 6 convolutional layers (tripling the depth). This additional capacity allows the network to better fit the complex representational targets discovered by search. To ensure fair comparison, we also test SGD baselines with 2, 4, and 6 convolutional layers per block  $(1 \times, 2 \times, 3 \times 3 \times 4)$  and report the best-performing configuration.

**Supervision variants.** By default, we supervise all three convolutional blocks with MSE losses and apply KL loss on the logits—we call this variant "Search-based (All layers)". Because earlier layers were difficult to optimize using search-based learning, we also tested an alternative approach: we skip direct supervision on convolutional block 0 and let it learn indirectly through gradients backpropagated from MSE losses in later blocks. We call this variant "Search-based (Skip block 0)". These variants are compared only in the results with data augmentation.

#### 4.3 RESULTS

To assess whether regressing to searched representations yields competitive generalization, we compare against standard stochastic gradient descent (SGD) training on MNIST, CIFAR-10, and CIFAR-100. Results are reported both with and without data augmentation to evaluate performance under minimal regularization. For MNIST, augmentation is omitted since performance already saturates without it. Since our search-based training does not use label smoothing, we also omit it from the SGD baselines for fair comparison. In the augmented setting, we cache each data point's searched representation once and reuse it across all augmented variants for efficiency.

Table 1 shows that without data augmentation, our method achieves test accuracy within 1% of SGD across all three benchmarks. With data augmentation (Table 2), the variant that supervises all layers (Search-based (All layers)) performs worse than SGD, trailing by a few points on CIFAR-10 and CIFAR-100. However, the variant that skips supervision on the first block (Search-based (Skip block 0)) performs substantially better—trailing SGD by just 1.0% on CIFAR-10 and 2.6% on CIFAR-100. This configuration allows the network body to still benefit from the searched representations, though the remaining gap indicates further improvements are needed. These results show that search-based regression can achieve competitive generalization with standard SGD training, even though it relies on cached representations.

Table 1: Test accuracies (%) without data augmentation. All results are reported as mean  $\pm$  std over 3 independent runs.

|                       | MNIST          | CIFAR-10       | CIFAR-100      |
|-----------------------|----------------|----------------|----------------|
| SGD                   | $99.1 \pm 0.1$ | $89.1 \pm 0.3$ | $62.3 \pm 0.5$ |
| Search-based Learning | $99.0 \pm 0.1$ | $88.3 \pm 0.3$ | $61.6 \pm 0.2$ |

Table 2: Test accuracies (%) with data augmentation. Mean  $\pm$  std over 3 runs. Search-based (Skip block 0) does not apply supervision to convolutional block 0; Search-based (All layers) supervises all blocks.

|                             | CIFAR-10       | CIFAR-100      |
|-----------------------------|----------------|----------------|
| SGD                         | $93.0 \pm 0.2$ | $71.8 \pm 0.4$ |
| Search-based (Skip block 0) | $92.0 \pm 0.1$ | $69.2 \pm 0.3$ |
| Search-based (All layers)   | $90.6 \pm 0.3$ | $66.5 \pm 0.3$ |

#### 4.4 EFFECTS OF SCALING SEARCH ON LEARNING

We saw in Section 3.2 that the fitness of searched representations improves with more search compute. Now, we investigate how the validation accuracy behaves when we train on representations obtained with increased search. Again, we scale search by varying the population size or number of generations while holding the other fixed. Then we run our training procedure on the cached representations formed by our search procedure.

We report the results in Figure 4. Observe that the accuracy scales with more compute along both the number of generations and the population size.

# 4.5 Comparing models trained with Search-based learning and SGD

The next natural question is whether the models trained with search-based learning are different from models trained with SGD. We intuitively expect this to be the case given the drastic difference between gradient-based learning in parameter space and search-based learning. Given the difficulty of measuring distance on models in parameter space, we instead measure the distance on the representations that trained models produce. We measure cosine distance during training on the validation set between the searched target representations and representations from models trained either with search-based learning or SGD. The cosine distance is computed on the activations after flattening the feature maps into vectors: a  $256 \times 15 \times 15$  feature map for block 0, a  $256 \times 7 \times 7$  feature map for block 1, and a  $256 \times 3 \times 3$  feature map for block 2. Cosine distance is one minus cosine similarity, ranging from 0 (identical) to 2 (opposite).

The results are shown in Figure 5. Across all three layers, the cosine distance between SGD and the searched targets remains large—close to 1.0—indicating that SGD converges to representational solutions that are quite different from those produced by search. At the same time, the plots show that the search-based training itself is well supervised across layers, with the cosine distance lowering over the course of training.

In addition to cosine distance, we also use collision entropy (defined in Section 3) to compare search-based learning and SGD. We calculate two types of collision entropy: within-class (averaging pairs of examples that belong to the same class) and between-class (averaging pairs of examples that belong to different classes). Lower values indicate more similar representations, while higher values indicate more distinct representations. This provides a complementary view of how the two training methods organize representation space.

Figure 6 shows the trajectories of within-class and between-class collision entropy during training at block 0. The patterns differ noticeably between search-based training and SGD. For example, representations from search-based training tend to form a more distinct separation in block 0 between classes. Plots for blocks 1 and 2 are included in Section B. These differences highlight that the learning dynamics of the two approaches are not the same, even though both produce meaningful organization of the representation space.

![](_page_8_Figure_0.jpeg)

Figure 4: Accuracy scales with compute usage in evolutionary search on CIFAR-100 (no data augmentation). Validation Accuracy vs. population size (left) and vs. generations (right).

![](_page_8_Figure_2.jpeg)

Figure 5: Cosine distance to searched representations vs. epoch at different searched layers with search-based training and SGD on CIFAR-100 (validation set, no data augmentation).

#### 5 RELATED WORK

Alternatives to backpropagation. Many alternatives to backpropagation focus on improving biological plausibility. Target propagation and its variants are the most similar to our method, which similarly does regression on layerwise targets (Lee et al., 2015; Bengio, 2014; Ernoult et al., 2022). Our method differs in that the targets are obtained through evolutionary search. The Forward-Forward method is an alternative to backpropagation which trains layers contrastively without any backward pass (Hinton, 2022). Feedback alignment is another alternative to backpropagation that uses fixed random feedback connections rather than exact gradient signals from transposed forward weights (Lillicrap et al., 2016). These alternatives to backpropagation change the learning mechanism for biological plausibility, unlike our method which aims to discover diverse models.

Search-based training methods. A parallel line of research uses search in neural net training rather than relying solely on gradient descent. Many neuroevolution methods search in parameter space. For example, Natural Evolution Strategies evolve a search distribution over parameters (Wierstra et al., 2014). NEAT takes a broader approach by searching over both parameters and network topologies (Stanley & Miikkulainen, 2002). Searching in parameter space can match gradient-based methods on certain reinforcement learning tasks (Salimans et al., 2017), and this method scales to networks with millions of parameters with massive parallelization (Such et al., 2017). However, these methods face computational challenges due to the high dimensionality of the parameter space. Other approaches avoid this limitation by searching in latent space. Latent Program Network searches in continuous latent program space with gradient descent (Macfarlane & Bonnet, 2024), and other work has also searched in a learned latent program space to maximize environment rewards (Trivedi et al., 2021). Our method performs search on the hidden activations of a network directly, rather than using a learned embedding space.

![](_page_9_Figure_0.jpeg)

Figure 6: Collision entropy within and between classes for SGD and search-based training (Block 0, validation set, no data augmentation). Plots for Blocks 1 and 2 are deferred to Appendix B. Experiments performed on CIFAR-100.

# 6 CONCLUSION, LIMITATIONS, AND FUTURE WORK

We demonstrated that neural network training can be decomposed into search over representations and gradient-based learning to match those representations. Our method achieves performance comparable to SGD while following fundamentally different optimization paths—the network body learns exclusively from searched representations rather than from classification gradients. This work provides a proof of concept that tractable search in representation space can guide parameter optimization, potentially addressing gradient descent's fundamental limitation of converging to a single solution.

Our approach has two main limitations. First, while our performance is comparable to SGD, we still trail behind it—our method is not yet a complete replacement for gradient descent. Future work must bridge this gap for the method to be viable in practice. Second, we currently use one-shot search with cached representations rather than iterative cycles where search and learning inform each other. Future work should implement tight feedback loops—networks learn searched representations, then trained networks inform the next search iteration. Additionally, while our search produces diverse representations (Section 3.2) that lead to distinct models compared to SGD (Section 4.5), more research is needed on how diverse representations translate to diverse model solutions—the ultimate goal.

# REFERENCES

Yoshua Bengio. How auto-encoders could provide credit assignment in deep networks via target propagation. *arXiv preprint arXiv:1407.7906*, 2014. doi: 10.48550/arXiv.1407.7906. URL https://arxiv.org/abs/1407.7906.

Maxence M Ernoult, Fabrice Normandin, Abhinav Moudgil, Sean Spinney, Eugene Belilovsky, Irina Rish, Blake Richards, and Yoshua Bengio. Towards scaling difference target propagation by learning backprop targets. In *International Conference on Machine Learning*, pp. 5968–5987. PMLR, 2022. URL https://proceedings.mlr.press/v162/ernoult22a.html.

Geoffrey Hinton. The forward-forward algorithm: Some preliminary investigations. *arXiv preprint arXiv:2212.13345*, 2022. doi: 10.48550/arXiv.2212.13345. URL https://arxiv.org/abs/2212.13345.

Dong-Hyun Lee, Saizheng Zhang, Asja Fischer, and Yoshua Bengio. Difference target propagation. In *Joint european conference on machine learning and knowledge discovery in databases*, pp. 498–515. Springer, 2015.

Timothy P Lillicrap, Daniel Cownden, Douglas B Tweed, and Colin J Akerman. Random synaptic feedback weights support error backpropagation for deep learning. *Nature communications*, 7:

- 13276, 2016. doi: 10.1038/ncomms13276. URL https://www.nature.com/articles/ncomms13276.
- Matthew V Macfarlane and Clément Bonnet. Searching latent program spaces. *arXiv preprint arXiv:2411.08706*, 2024. URL https://arxiv.org/abs/2411.08706.
- Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, and Ilya Sutskever. Evolution strategies as a scalable alternative to reinforcement learning. *arXiv preprint arXiv:1703.03864*, 2017. URL https://arxiv.org/abs/1703.03864.
- Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Nikul Patel, Jalal Naghiyev, Yann LeCun, and Ravid Shwartz-Ziv. Layer by layer: Uncovering hidden representations in language models. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=WGXb7UdvTX.
- Ray J Solomonoff. A formal theory of inductive inference. part i. *Information and control*, 7(1): 1–22, 1964.
- Kenneth O. Stanley and Risto Miikkulainen. Evolving neural networks through augmenting topologies. *Evolutionary Computation*, 10(2):99–127, 2002. doi: 10.1162/106365602320169811. URL https://direct.mit.edu/evco/article/10/2/99/1123.
- Felipe Petroski Such, Vashisht Madhavan, Edoardo Conti, Joel Lehman, Kenneth O. Stanley, and Jeff Clune. Deep neuroevolution: Genetic algorithms are a competitive alternative for training deep neural networks for reinforcement learning. *arXiv preprint arXiv:1712.06567*, 2017. URL https://arxiv.org/abs/1712.06567.
- Dweep Trivedi, Jesse Zhang, Shao-Hua Sun, and Joseph J. Lim. Learning to synthesize programs as interpretable and generalizable policies. In *Advances in Neural Information Processing Systems*, 2021. URL https://arxiv.org/abs/2108.13643.
- Kai Wang, xu Zhao Pan, Zhuang Liu, Zelin Zang, Trevor Darrell, and Yang You. Neural network diffusion, 2024. URL https://openreview.net/forum?id=8Q6UmFhhQS.
- Daan Wierstra, Tom Schaul, Tobias Glasmachers, Yi Sun, Jan Peters, and Jürgen Schmidhuber. Natural evolution strategies. *Journal of Machine Learning Research*, 15(27):949–980, 2014. URL https://www.jmlr.org/papers/v15/wierstra14a.html.

#### A IMPLEMENTATION DETAILS

#### A.1 MODEL ARCHITECTURE

We adapt the optimized CIFAR architecture from https://github.com/KellerJordan/cifar10-airbench with two changes: (1) first block widened from 64 to 256 channels, (2) final max pool removed in favor of a larger linear layer.

**Overall structure:** Whitening layer  $\to 3$  convolutional blocks  $\to$  linear classifier. The blocks produce feature maps of  $256 \times 15 \times 15$ ,  $256 \times 7 \times 7$ , and  $256 \times 3 \times 3$  respectively.

Whitening:  $3 \rightarrow 24$  channels, kernel size 2, learned via eigendecomposition of 5000 training patches.

 $\begin{array}{l} \textbf{ConvGroup structure:} \ Conv(3\times3) \rightarrow MaxPool(2\times2) \rightarrow BatchNorm \rightarrow GELU \rightarrow Conv(3\times3) \\ \rightarrow BatchNorm \rightarrow GELU. \end{array}$ 

**Convolutional blocks:** Each of the 3 convolutional blocks consists of:

- 1 ConvGroup with pooling (reduces spatial dimensions)
- 2 additional ConvGroups without pooling (increases capacity for learning searched representations)

This gives each block 3 ConvGroups total, tripling the capacity compared to a single ConvGroup baseline.

**Classification head:** Linear layer mapping flattened features to class logits, with output scaling factor 1.0.

## A.2 SEARCH HYPERPARAMETERS

**Population structure:** 240 total candidates, selecting 20 parents per image, generating 100 exploratory children (high mutation) and 120 refinement children (standard mutation).

#### **Evolution schedule:**

- Layer 0: 300 generations, mutation strength  $\alpha = 0.3$ , exploration boost  $5.0 \times 5$  blur passes
- Layer 1: 100 generations,  $\alpha = 0.2$ , exploration boost  $3.0 \times$ , 1 blur pass
- Layer 2: 100 generations,  $\alpha = 0.1$ , exploration boost  $3.0 \times$ , 1 blur pass
- Logits: 10 generations,  $\alpha = 0.1$ , exploration boost  $1.0 \times$ , no blur

**Selection:** Top-k selection performed independently per image rather than across the batch.

#### A.3 TRAINING HYPERPARAMETERS

**Optimization:** SGD with learning rate 4.0, momentum 0.85 with Nesterov acceleration.

**Loss weights:** MSE coefficient 1.0 for each convolutional layer, KL divergence coefficient  $\lambda=0.03$  for logits (with stop-gradient preventing backpropagation through the network body).

**Batching:** Batch size 2000 with deterministic batch composition across epochs.

**Network regularization:** BatchNorm momentum 0.6, no running statistics. No label smoothing (our search-based training does not use label smoothing, so we disable it for SGD for fairness).

**Data augmentation:** Random horizontal flips and random crops from images padded by 2 pixels.

**Validation strategy:** Hold out 4000 training samples (8% of 50k) for model selection during training, evaluating the best model on the test set only at the end.

**Training duration:** 90 epochs.

## A.4 REPRODUCTION OPERATOR DETAILS

This section provides the mathematical specifications for the genetic operators described in Section 3.1.

**Crossover.** For two randomly selected parents  $H_{p_1}^{(\ell)}$  and  $H_{p_2}^{(\ell)}$  at layer  $\ell$ :

$$\tilde{H}^{(\ell)} = \frac{1}{2} (H_{p_1}^{(\ell)} + H_{p_2}^{(\ell)})$$

Mutation. The mutation process differs between convolutional layers and logits.

Convolutional layers ( $\ell < 3$ ): Generate channel-wise Gaussian noise  $\epsilon_{b,c,h,w} \sim \mathcal{N}(0,(\alpha\sigma_{b,c})^2)$ , where  $\sigma_{b,c}$  is the average standard deviation of the two parents for batch element b and channel c. Apply mutation:  $\tilde{H}^{(\ell)} = \tilde{H}^{(\ell)} + \epsilon$ .

**Logits layer** ( $\ell=3$ ): Generate element-wise Gaussian noise  $\epsilon_{b,k} \sim \mathcal{N}(0,(\alpha\sigma_b)^2)$ , where  $\sigma_b$  is the average standard deviation of the two parents across classes. Apply mutation and recenter:  $\tilde{H}^{(\ell)} = \tilde{H}^{(\ell)} + \epsilon - \text{mean}(\tilde{H}^{(\ell)})$ .

**Spatial Smoothing (Convolutional layers only).** Apply repeated  $3 \times 3$  average pooling with padding to reduce high-frequency artifacts.

**Normalization (Convolutional layers only).** After mutation, normalize each sample to zero mean and unit variance:

$$\tilde{H}_{b,:,:,:}^{(\ell)} \leftarrow \frac{\tilde{H}_{b,:,:,:}^{(\ell)} - \mu_b}{\sqrt{v_b + \epsilon}}$$

where  $\mu_b$  and  $v_b$  are the mean and variance computed over channels and spatial dimensions for batch element b, and  $\epsilon = 10^{-5}$  for numerical stability.

## A.5 MNIST HYPERPARAMETERS

We use the same search configuration as CIFAR-10 with one modification: we reduce mutation strength to  $\alpha=0.1$  uniformly across all layers (layers 0–2 and logits), compared to the layer-specific values used for CIFAR. We also increase the training duration to 1000 epochs. During learning, the MNIST model uses four convolutional layers (vs. six for CIFAR). All other hyperparameters remain identical to the CIFAR configuration.

## B ADDITIONAL COLLISION ENTROPY PLOTS

![](_page_13_Figure_0.jpeg)

Figure 7: Collision entropy within and between classes for SGD and search-based training at Block 1 and Block 2 on CIFAR-100 (validation set, no data augmentation).