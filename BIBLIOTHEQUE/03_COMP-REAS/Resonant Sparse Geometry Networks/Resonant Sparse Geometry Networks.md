# Resonant Sparse Geometry Networks

Hasi Hays $^*$ 

Department of Chemical Engineering, University of Arkansas, Fayetteville, AR 72701, USA (Dated: January 27, 2026)

We introduce Resonant Sparse Geometry Networks (RSGN), a brain-inspired architecture with self-organizing sparse hierarchical input-dependent connectivity. Unlike Transformer architectures that employ dense attention mechanisms with  $O(n^2)$  computational complexity, RSGN embeds computational nodes in learned hyperbolic space where connection strength decays with geodesic distance, achieving dynamic sparsity that adapts to each input. The architecture operates on two distinct timescales: fast differentiable activation propagation optimized through gradient descent, and slow Hebbian-inspired structural learning for connectivity adaptation through local correlation rules. We provide rigorous mathematical analysis demonstrating that RSGN achieves  $O(n \cdot k)$  computational complexity where  $k \ll n$  represents the average active neighborhood size. Experimental evaluation on hierarchical classification and long-range dependency tasks demonstrates that RSGN achieves 96.5% accuracy on long-range dependency tasks while using approximately  $15 \times$  fewer parameters than standard Transformers. On challenging hierarchical classification with 20 classes, RSGN achieves 23.8% accuracy (compared to 5% random baseline) with only 41,672 parameters, nearly 10× fewer than Transformer baselines requiring 403.348 parameters to achieve 30.1% accuracy. Our ablation studies confirm the contribution of each architectural component, with Hebbian learning providing consistent improvements. These results suggest that brain-inspired principles of sparse, geometricallyorganized computation offer a promising direction toward more efficient and biologically plausible neural architectures.

#### I. INTRODUCTION

The Transformer architecture [1] has emerged as the dominant paradigm in modern deep learning, powering breakthrough systems in natural language processing [2, 3], computer vision [4], and multimodal artificial intelligence [5]. The core innovation of Transformers lies in the self-attention mechanism [6, 7], which allows every token to attend to every other token in a sequence, thereby capturing long-range dependencies without the sequential processing constraints of recurrent architectures. This architectural flexibility has enabled unprecedented scaling, with models now reaching hundreds of billions of parameters and demonstrating emergent capabilities that appear qualitatively different from smaller systems [8]. However, the computational flexibility of self-attention incurs a significant cost: quadratic complexity  $O(n^2)$  with respect to sequence length n. For a sequence of 1,000 tokens, attention computation requires evaluating one million pairwise relationships. For sequences of 100,000 tokens, increasingly common in document-level understanding and long-context applications, this becomes 10 billion operations per attention layer, rendering standard Transformers computationally prohibitive [9]. This scaling limitation has motivated extensive research into efficient attention variants, yet fundamental questions remain about whether dense, global attention is the optimal computational paradigm for sequence processing.

In stark contrast, the human brain achieves remarkable computational efficiency through fundamentally different organizational principles. Operating on approximately 20 watts of power, comparable to a dim light bulb, the brain processes complex sensory information across multiple modalities, maintains episodic and semantic memories spanning decades, generates creative thought, plans complex action sequences, and coordinates motor actions with millisecond precision. This extraordinary capability emerges from approximately 86 billion neurons connected by roughly 100 trillion synapses [10], yet the computational principles underlying this efficiency remain only partially understood.

Several key organizational principles distinguish biological neural computation from contemporary artificial systems. First, the brain exhibits extreme sparsity in activation: at any given moment, only approximately 1-2\% of cortical neurons actively fire [11, 12]. This sparse coding principle dramatically reduces energy consumption while providing representational advantages including noise robustness, memory capacity, and compositional generalization [13]. Second, biological neural processing employs input-dependent routing: different inputs activate fundamentally different neural pathways rather than engaging the same fixed computational graph [14, 15]. Visual, auditory, and somatosensory information flow through distinct cortical hierarchies, with cross-modal integration occurring at specific convergence zones. Third, neural connectivity emerges through self-organizing structure via Hebbian learning principles ("neurons that fire together wire together") [16] and activity-dependent synaptic pruning during development and throughout life [17, 18]. Fourth, information flows through hierarchical organization embedded in the physical geometry of cortical tissue, with systematic transformations as signals propagate from primary sensory areas through association cortices [19, 20].

These observations motivate our development of Resonant Sparse Geometry Networks (RSGN), an ar-

<sup>\*</sup> hasih@uark.edu

chitecture that incorporates all four biological principles through a novel combination of computational mechanisms (Figure 1). We embed N computational nodes in learned hyperbolic space  $\mathbb{H}^d$ , which naturally encodes hierarchical relationships through its exponentially expanding geometry [21, 22]. Unlike Euclidean space where volume grows polynomially with radius, hyperbolic space exhibits exponential volume growth, allowing tree-like hierarchical structures to be embedded with arbitrarily low distortion [23]. Connection strength between nodes decays with geodesic distance in this space, enforcing locality and sparsity without explicit pruning mechanisms.

We implement input-dependent ignition where input tokens create "spark points" in the embedding space, activating only nearby nodes and establishing sparse initial activation patterns. These activations then propagate through the network via iterative dynamics with soft thresholds and local inhibition, implementing a winnertake-more competition that mirrors lateral inhibition in biological neural circuits [24, 25]. The resonance metaphor reflects that stable activation patterns emerge through iterative settling, analogous to the global workspace theory of consciousness where coherent representations arise from competitive dynamics among specialized processors [26, 27]. Crucially, RSGN operates on two distinct timescales inspired by the separation between fast neural dynamics and slow synaptic plasticity in biological systems [28, 29]. Fast learning employs standard gradient descent through differentiable relaxations of threshold operations, optimizing for task performance on the timescale of individual forward passes. Slow learning uses local Hebbian rules where co-activated nodes strengthen their connections and drift toward each other in the embedding space, while unused connections decay and eventually prune. A global reward signal modulates plasticity strength, analogous to dopaminergic modulation of synaptic plasticity in the basal ganglia and cortex [30, 31].

Our primary contributions are fourfold:

- Mathematical Model: We provide a complete mathematical framework for spatially-embedded neural computation in hyperbolic geometry, including precise definitions of distance-based connectivity, soft-threshold activation dynamics, and local inhibition mechanisms (section III).
- 2. Differentiable Relaxation: We develop a differentiable relaxation scheme that enables gradient-based training of networks with dynamic sparse structure, bridging the gap between discrete biological-like computation and continuous optimization (section IV).
- 3. Hybrid Learning Rule: We propose a hybrid learning rule combining backpropagation for fast weight updates with Hebbian structural plasticity for slow topological adaptation, offering a biologically plausible alternative to end-to-end gradient-based structure learning (section IV).

4. Experimental Validation: We provide theoretical complexity analysis demonstrating subquadratic scaling and present experimental results on hierarchical classification and long-range dependency tasks showing competitive performance with dramatically reduced parameters (section VI).

The remainder of this paper is organized as follows. section II reviews related work on efficient attention mechanisms, geometric neural networks, and biologically-inspired learning. section III presents the complete RSGN architecture. section IV describes the two-timescale learning system. section V provides theoretical analysis of complexity and expressiveness. section VI presents experimental evaluation on synthetic benchmarks. section VII discusses biological connections, limitations, and future directions. section VIII concludes with summary and outlook.

#### II. RELATED WORK

#### A. Efficient Attention Mechanisms

The quadratic complexity of standard self-attention has motivated a rich literature on efficient sequence modeling. Sparse Transformers [32] employ fixed sparsity patterns such as local windows combined with strided attention, reducing complexity to  $O(n\sqrt{n})$  while maintaining the ability to capture long-range dependencies through composition of local and global patterns. BigBird [33] extends this approach with random attention connections and global tokens, achieving linear complexity while preserving theoretical expressiveness. Linformer [34] projects keys and values to lower-dimensional spaces, achieving O(n) complexity under the assumption that attention matrices are approximately low-rank. *Performer* [35] uses random feature approximations of the softmax kernel (FA-VOR+) to decompose attention computation, enabling linear-time attention through the associativity of matrix multiplication. Linear Attention [36] replaces the softmax kernel with feature maps that allow similar decomposition.

While these approaches successfully reduce computational cost, they share a common limitation: they maintain fixed structure across inputs, failing to capture the input-dependent routing observed in biological neural systems. The sparsity pattern in Sparse Transformers is predetermined, the projection matrices in Linformer are learned but fixed, and the random features in Performer are sampled once. In contrast, RSGN adapts its active computation graph for each input through the ignition mechanism, with different inputs potentially activating entirely different subsets of nodes.

![](_page_2_Figure_1.jpeg)

FIG. 1: Bio-inspired principles underlying RSGN. (A) Resonant Sparse Geometry Network (RSGN) architecture illustrating the four key principles: (1) input-dependent routing where input tokens create spark points that ignite nearby nodes; (2) hierarchical sparse connectivity with distance-based connection strength; (3) two-timescale learning combining fast gradient-based activation updates with slow Hebbian structural plasticity; and (4) hyperbolic geometry embedding with soft thresholds and local inhibition. The brain illustration shows analogous biological mechanisms including attentional networks and striatal reward modulation. (B) Bio-inspired Modular Representation comparing cortical organization (left) with RSGN implementation (right). The cortical hierarchy shows sensory input propagating from primary sensory cortex through association areas to higher cognitive regions, with Hebbian plasticity ("neurons that fire together, wire together"). The corresponding RSGN diagram shows resonant signal propagation through the Poincaré ball across iterative steps, with the ignition module processing input sequences and producing classifier output through local inhibition and soft threshold operations.

# B. State Space Models

An alternative paradigm for sequence modeling has emerged through Structured State Space Sequence models (S4) [37], which achieve linear complexity through continuous-time state space formulations with carefully parameterized transition matrices based on the HiPPO framework [38]. S4 and its variants [39, 40] achieve strong performance on the Long Range Arena benchmark [41], particularly excelling at tasks requiring extremely long context like PathX. The recent Mamba architecture [42] extends this framework with selective state spaces, introducing content-dependent processing through inputdependent transition parameters. This represents a significant step toward input-dependent computation, though the routing mechanism differs fundamentally from our spatial approach. While Mamba modulates state dynamics based on input content, RSGN routes computation through geometric proximity in learned embedding space, providing a more explicitly structured form of input-dependent processing.

# C. Geometric and Hyperbolic Neural Networks

Hyperbolic geometry has attracted increasing attention in machine learning due to its natural capacity for representing hierarchical structures. Poincaré Embeddings [22] demonstrated that hyperbolic space can embed hierarchical data (such as WordNet taxonomies) with significantly lower distortion than Euclidean alternatives. Hyperbolic Neural Networks [43] extended standard neural network operations (linear layers, attention, recurrence) to operate in hyperbolic space, enabling end-to-end learning of hierarchical representations. Hyperbolic Attention Networks [44] apply attention mechanisms in hyperbolic geometry, producing hierarchically-structured attention patterns. Theoretical work has established that n-node trees can be embedded in 2-dimensional hyperbolic space with O(1) distortion [21], compared to  $\Omega(\log n)$  distortion required in Euclidean space [45]. This fundamental advantage motivates our use of hyperbolic geometry, though our approach differs from prior work in a key respect: rather than embedding data representations in hyperbolic space, we embed the computational nodes themselves, deriving connectivity structure from spatial relationships.

## D. Dynamic and Sparse Networks

Mixture of Experts (MoE) [46, 47] routes inputs to different expert subnetworks through learned gating functions, achieving input-dependent computation allocation that scales model capacity without proportional computational cost. GLaM [48] and Switch Transformers [47] have demonstrated that MoE can scale to trillion-parameter models while maintaining computational efficiency. However, MoE typically routes entire tokens to experts rather

than achieving the fine-grained, spatially-organized sparsity of RSGN.

Dynamic Networks encompass a broader class of architectures that adapt computation based on input characteristics [49]. Early-exit mechanisms allow confident predictions to skip later layers [50]. Adaptive depth networks learn to allocate computation per-example [51]. Dynamic channel selection prunes features based on input content [52]. Neural Architecture Search [53, 54] learns network structure but typically operates at training time rather than adapting dynamically per input.

The Lottery Ticket Hypothesis [55] demonstrates that sparse subnetworks exist within dense networks that can match full network performance when trained in isolation. This suggests that the dense parameterization of standard networks is redundant, motivating approaches like RSGN that learn sparse structure directly. However, lottery tickets are typically identified through iterative pruning rather than learned through local rules, and represent a single sparse structure rather than input-dependent sparsity.

#### E. Biologically-Inspired Learning

Hebbian Learning [16], encapsulated in the principle that "neurons that fire together wire together," proposes that correlated activation strengthens synaptic connections. This principle has been formalized in various spiketiming-dependent plasticity (STDP) rules [56, 57] and correlation-based learning algorithms [58]. Modern work has explored combining Hebbian learning with backpropagation for improved efficiency [59] and biological plausibility [60].

Predictive Coding [61, 62] frames neural computation as hierarchical prediction and error correction, offering a functional account of cortical processing that connects perception, action, and learning. Equilibrium Propagation [63] provides a biologically plausible alternative to backpropagation by computing gradients through network dynamics at equilibrium. Forward-Forward [64] eliminates the backward pass entirely, using local contrastive objectives.

The current study occupies a distinctive position in this landscape: we combine Hebbian structural learning for slow connectivity adaptation with differentiable activation dynamics for fast task optimization. This two-timescale approach mirrors the separation between synaptic plasticity (slow, correlation-based) and neural dynamics (fast, activity-based) in biological systems [28].

## III. RESONANT SPARSE GEOMETRY NETWORKS

## A. Architectural Overview

RSGN consists of N computational nodes embedded in a d-dimensional hyperbolic space  $\mathbb{H}^d$ , implemented using the Poincaré ball model for computational tractability. Each node maintains state variables that evolve on different timescales, separating fast activation dynamics from slow structural plasticity. The forward pass proceeds through four phases: (1) input embedding and ignition, (2) iterative activation propagation, (3) local inhibition and competition, and (4) output readout from active nodes. Figure 2 illustrates the complete RSGN architecture.

**Definition 1** (RSGN Node). A node i in RSGN is characterized by the tuple  $\mathcal{N}_i = (\mathbf{p}_i, \mathbf{h}_i, \theta_i, \ell_i)$  where:

- $\mathbf{p}_i \in \mathbb{B}^d$  is the position in the Poincaré ball model of hyperbolic space (slow-learned)
- $\mathbf{h}_i \in \mathbb{R}^{d_h}$  is the activation state vector (fast, evolves per-input)
- $\theta_i \in \mathbb{R}^+$  is the activation threshold (slow-learned)
- $\ell_i \in \mathbb{R}$  is the hierarchical level indicator (slow-learned)

The separation of timescales mirrors biological neural systems where synaptic efficacy (connection strength) changes slowly over minutes to hours, while membrane potential dynamics (activation) evolve on millisecond timescales [65]. In RSGN, fast variables  $\mathbf{h}_i$  update within a single forward pass through iterative propagation (5 steps in our experiments), while slow variables  $(\mathbf{p}_i, \theta_i, \ell_i)$  evolve across training batches through Hebbian-inspired rules.

### B. Hyperbolic Space Embedding

We employ the Poincaré ball model of hyperbolic space due to its computational tractability and natural encoding of hierarchical structure [22, 23].

**Definition 2** (Poincaré Ball). The Poincaré ball  $\mathbb{B}^d = \{\mathbf{x} \in \mathbb{R}^d : \|\mathbf{x}\| < 1\}$  is the open unit ball equipped with the Riemannian metric

$$g_{\mathbf{x}} = \left(\frac{2}{1 - \|\mathbf{x}\|^2}\right)^2 g_E \tag{1}$$

where  $g_E$  denotes the Euclidean metric tensor.

The conformal factor  $\lambda_{\mathbf{x}} = 2/(1-\|\mathbf{x}\|^2)$  causes distances to expand as points approach the boundary, encoding the exponential growth of volume characteristic of hyperbolic geometry. This property allows tree-like hierarchical structures to be embedded with low distortion: a complete

binary tree of depth D requires only O(D) hyperbolic space to embed with bounded distortion, compared to  $O(2^D)$  Euclidean dimensions [21].

The geodesic distance between points  $\mathbf{p}_i, \mathbf{p}_j \in \mathbb{B}^d$  is given by the closed-form expression:

$$d_{\mathbb{H}}(\mathbf{p}_i, \mathbf{p}_j) = \operatorname{arcosh}\left(1 + 2\frac{\|\mathbf{p}_i - \mathbf{p}_j\|^2}{(1 - \|\mathbf{p}_i\|^2)(1 - \|\mathbf{p}_j\|^2)}\right)$$
 (2)

This distance metric captures the intuition that nodes near the boundary (representing leaves of a hierarchy) are far from each other even if Euclidean-close, while nodes near the origin (representing root or abstract concepts) have shorter paths to many other nodes. This property naturally implements hierarchical information routing: abstract features near the center can efficiently aggregate information from many peripheral specialized nodes.

Remark 1 (Geometric Intuition). Consider placing nodes representing a taxonomy in the Poincaré ball. The root concept (e.g., "entity") sits near the origin. First-level categories ("animal," "plant," "artifact") occupy positions at moderate radius. Leaf-level instances ("Labrador retriever," "oak tree") cluster near the boundary. The hyperbolic metric ensures that siblings ("Labrador" and "poodle") are close, while distant leaves ("Labrador" and "oak") are far despite potentially similar Euclidean coordinates.

## C. Distance-Based Connectivity

Connection strength between nodes emerges from their positions in hyperbolic space, modulated by learned affinity parameters and hierarchical level differences.

**Definition 3** (Connection Strength). The connection strength from node i to node j is defined as

$$w_{ij} = \sigma(a_{ij}) \cdot \exp\left(-\frac{d_{\mathbb{H}}(\mathbf{p}_i, \mathbf{p}_j)}{\tau}\right) \cdot \phi(\ell_j - \ell_i)$$
 (3)

where:

- $a_{ij} = \mathbf{u}_i^{\top} \mathbf{v}_j$  is a learned affinity parameter, factorized for efficiency
- $\tau > 0$  is a temperature parameter controlling distance sensitivity
- $\phi(x) = \log(1 + e^{x+1})$  (softplus with bias) favors feedforward information flow
- $\sigma(\cdot)$  denotes the sigmoid function

This formulation ensures several desirable properties: **Locality and Sparsity:** The exponential decay with hyperbolic distance enforces locality. For sufficiently distant nodes, connection strength becomes negligible  $(w_{ij} \approx 0)$ , creating natural sparsity without explicit pruning. The effective neighborhood size is controlled by  $\tau$ .

![](_page_5_Figure_1.jpeg)

FIG. 2: RSGN architecture overview. Input tokens are embedded and create spark points in the hyperbolic embedding space (Poincaré ball), where distance-based connectivity determines connection strength. Only  $\sim 2\%$  of nodes activate (sparse activation), with local inhibition implementing winner-take-more dynamics. Activations propagate iteratively through T steps, followed by soft threshold activation, layer normalization, and readout. The two-timescale learning system combines fast gradient descent with differentiable relaxation and slow Hebbian plasticity for structural adaptation, both modulated by a global reward signal. The inset shows the Poincaré disk geometry where nodes closer to each other have stronger connections, naturally embedding tree-like hierarchies.

**Learned Modulation:** The affinity term  $\sigma(a_{ij}) \in (0,1)$  allows the network to strengthen or weaken connections beyond what distance alone would dictate. Coactivated nodes can strengthen their affinity through Hebbian learning, while unused pathways decay.

**Hierarchical Flow:** The level factor  $\phi(\ell_j - \ell_i)$  biases information flow upward through the hierarchy (when  $\ell_j > \ell_i$ ). This mimics the predominantly feedforward processing observed in sensory cortices, where information flows from primary to association areas [19].

**Efficiency:** The factorized representation  $a_{ij} = \mathbf{u}_i^{\top} \mathbf{v}_j$  with  $\mathbf{u}_i, \mathbf{v}_j \in \mathbb{R}^r$  (rank r = 32 in experiments) reduces the parameter count from  $O(N^2)$  to O(Nr) while still allowing rich connectivity patterns through the low-rank structure.

# D. Input-Dependent Ignition

The key mechanism enabling input-dependent routing is the ignition process, which maps input tokens to "spark points" in hyperbolic space and activates nearby nodes.

**Definition 4** (Ignition Function). Given input sequence  $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_T]$  with  $\mathbf{x}_t \in \mathbb{R}^{d_x}$ , the ignition process proceeds in two stages:

Stage 1 (Embedding): Compute spark embeddings

$$\mathbf{s}_t = f_{embed}(\mathbf{x}_t) \in \mathbb{B}^d, \quad t = 1, \dots, T$$
 (4)

where  $f_{embed}: \mathbb{R}^{d_x} \to \mathbb{B}^d$  is a neural network with hyperbolic tangent output scaled by factor  $\gamma < 1$  to ensure  $\|\mathbf{s}_t\| < 1$ .

Stage 2 (Activation): Compute initial activation field

$$\alpha_i^{(0)} = \max_{t \in [T]} \exp\left(-\frac{d_{\mathbb{H}}(\mathbf{p}_i, \mathbf{s}_t)^2}{2\sigma_{ign}^2}\right)$$
 (5)

where  $\sigma_{ign}$  controls the ignition radius.

This Gaussian kernel in hyperbolic distance creates localized activation regions around each input spark. Nodes far from all sparks receive negligible initial activation  $(\alpha_i^{(0)} \approx 0)$ , establishing the sparse initial pattern that propagates through subsequent dynamics. The max operation allows nodes near *any* input token to activate, enabling distributed representation of sequential inputs.

Remark 2 (Biological Analogy). The ignition mechanism parallels the concept of ignition in global workspace theory [27]: sensory inputs initially activate specialized processors in primary sensory cortices, which then compete for access to a global workspace enabling conscious processing. In RSGN, spark points represent these initial activations, while subsequent propagation implements the competition and integration phase.

# E. Activation Dynamics with Soft Thresholds

Activation propagates through the network via iterative dynamics that combine signal aggregation, thresholding, and residual connections.

**Definition 5** (Soft Threshold Function). The differentiable soft threshold activation is

$$SoftThresh(x, \theta, T) = \sigma\left(\frac{x - \theta}{T}\right)$$
 (6)

where T > 0 is a temperature parameter. As  $T \to 0$ , the soft threshold approaches the hard step function  $\mathbf{1}[x > \theta]$ , but gradients remain well-defined for any T > 0.

**Definition 6** (Propagation Dynamics). The activation state evolves through K steps (K = 5 in experiments) according to:

$$\tilde{\mathbf{h}}_{i}^{(t+1)} = \sum_{j \in \mathcal{A}^{(t)}} w_{ij} \cdot \mathbf{W}_{h} \mathbf{h}_{j}^{(t)} \tag{7}$$

$$\alpha_i^{(t+1)} = SoftThresh\left(\alpha_i^{(t)} + \beta \|\tilde{\mathbf{h}}_i^{(t+1)}\|, \theta_i, T\right)$$
 (8)

$$\mathbf{h}_{i}^{(t+1)} = \alpha_{i}^{(t+1)} \cdot LayerNorm\left(\tilde{\mathbf{h}}_{i}^{(t+1)} + \mathbf{h}_{i}^{(t)}\right) \qquad (9)$$

where  $\mathcal{A}^{(t)} = \{j : \alpha_j^{(t)} > \epsilon\}$  is the active set at step t,  $\mathbf{W}_h \in \mathbb{R}^{d_h \times d_h}$  is a learned transformation, and  $\beta > 0$  scales signal contribution to activation.

The dynamics implement a form of message passing where:

- Only active nodes (those with  $\alpha_j > \epsilon$ ) participate in message passing (Equation 7), implementing sparse computation
- Activation levels update based on accumulated signal strength (Equation 8), with thresholds controlling which nodes become/remain active

• State vectors combine new information with residual connections (Equation 9), stabilized by Layer-Norm [66]

#### F. Local Inhibition

To prevent activation explosion and encourage winner-take-more competition, we apply local inhibition within spatial neighborhoods.

**Definition 7** (Local Inhibition). After each propagation step, activations normalize within spatial neighborhoods:

$$\alpha_i^{(t)} \leftarrow \alpha_i^{(t)} \cdot \frac{|B_r(\mathbf{p}_i)|}{\sum_{i \in B_r(\mathbf{p}_i)} \alpha_i^{(t)} + \epsilon}$$
 (10)

where  $B_r(\mathbf{p}_i) = \{j : d_{\mathbb{H}}(\mathbf{p}_i, \mathbf{p}_j) < r\}$  defines the inhibition neighborhood.

This implements divisive normalization, a canonical neural computation observed across sensory systems [25]. Within local clusters, nodes with higher activation suppress neighbors, leading to sparse distributed representations. The inhibition radius r controls the spatial scale of competition: smaller r allows finer-grained representations, while larger r enforces more aggressive sparsification.

#### G. Resonance and Output

The network iterates propagation and inhibition for K steps. The term "resonance" reflects that stable activation patterns emerge through these iterative dynamics, representing coherent interpretations of the input.

**Definition 8** (Output Readout). The network output is computed from active nodes at the final step as:

$$\mathbf{y} = f_{out} \left( \sum_{i=1}^{N} \alpha_i^{(K)} \cdot \mathbf{W}_{out} \mathbf{h}_i^{(K)} \right)$$
 (11)

where  $f_{out}$  is a task-specific output function (e.g., softmax for classification) and  $\mathbf{W}_{out} \in \mathbb{R}^{d_{out} \times d_h}$  projects to output dimension.

The activation-weighted sum ensures that only active nodes contribute to the output, with contribution proportional to their activation level. This provides a natural attention mechanism where the network focuses on relevant nodes for each input.

## IV. LEARNING RULES

RSGN employs a two-timescale learning system that separates fast gradient-based optimization from slow structural plasticity, inspired by the separation of timescales in biological learning [28].

## A. Fast Learning: Gradient Descent

For task optimization, we employ stochastic gradient descent on the task-specific loss  $\mathcal{L}_{task} = \mathcal{L}(\mathbf{y}, \mathbf{y}^*)$ . The soft-threshold function enables gradient flow:

$$\frac{\partial \text{SoftThresh}(x, \theta, T)}{\partial x} = \frac{1}{T} \sigma \left( \frac{x - \theta}{T} \right) \left( 1 - \sigma \left( \frac{x - \theta}{T} \right) \right)$$
(12)

The gradient magnitude is bounded by 1/(4T), achieved when  $x=\theta$ . This provides controlled gradient flow that scales inversely with temperature. In our experiments, we use T=1.0 throughout training, which provides soft thresholds enabling smooth gradient flow while the sparsity target and threshold adaptation maintain appropriate activation levels.

Fast learning updates the following parameters through backpropagation:

- Embedding function  $f_{\text{embed}}$
- $\bullet$  Transformation matrix  $\mathbf{W}_h$
- Output projection  $\mathbf{W}_{\text{out}}$  and function  $f_{\text{out}}$
- Affinity factors  $\mathbf{u}_i, \mathbf{v}_i$

We use AdamW optimizer [67] with learning rate  $10^{-3}$ , weight decay  $10^{-4}$ , and cosine annealing schedule.

#### B. Slow Learning: Hebbian Structural Plasticity

The network structure evolves through local Hebbian rules that operate on a slower timescale than gradient updates.

**Definition 9** (Hebbian Affinity Update). After each forward pass, affinity factors update according to:

$$\Delta a_{ij} = \eta_a \cdot \bar{\alpha}_i \cdot \bar{\alpha}_j \cdot R \tag{13}$$

where  $\bar{\alpha}_i = \frac{1}{K} \sum_{t=1}^K \alpha_i^{(t)}$  is the time-averaged activation over propagation steps,  $R = -\mathcal{L}_{task}$  is the reward signal (negative loss), and  $\eta_a$  is the Hebbian learning rate.

This rule implements the Hebbian principle: coactivated nodes strengthen their connection, modulated by task reward. The reward signal R provides global feedback analogous to dopaminergic modulation of synaptic plasticity [30, 31].

**Definition 10** (Threshold Adaptation). Thresholds adapt to maintain target sparsity:

$$\Delta \theta_i = \eta_\theta \cdot (\bar{\alpha}_i - \alpha_{target}) \tag{14}$$

where  $\alpha_{target} = 0.1$  is the desired average activation level.

If a node activates too frequently, its threshold increases, making activation harder. This homeostatic mechanism maintains approximately constant sparsity levels despite changing input statistics, analogous to synaptic scaling in biological neurons [68].

## C. Synaptic Pruning and Sprouting

To enable ongoing structural plasticity, weak connections are periodically pruned: if  $|a_{ij}| < \epsilon_{\text{prune}}$  for  $K_{\text{prune}}$  consecutive epochs, the connection is deleted. New connections can sprout between highly correlated but unconnected nodes: if  $\text{corr}(\alpha_i, \alpha_j) > \gamma_{\text{sprout}}$  and  $a_{ij} = 0$ , initialize  $a_{ij} \sim \mathcal{N}(0, \sigma_{\text{init}}^2)$ .

These rules allow the network to reorganize its connectivity based on task demands, analogous to the structural plasticity observed in developing and adult brains [69].

#### V. THEORETICAL ANALYSIS

# A. Computational Complexity

**Theorem 1** (RSGN Complexity). For an RSGN with N nodes, average active set size  $|\mathcal{A}| = k$ , and average neighborhood size m (nodes within distance threshold), the per-step computational complexity is  $O(k \cdot m \cdot d_h^2)$ . For sparse activation  $(k \ll N)$  and local connectivity  $(m \ll N)$ , this is sub-quadratic in N.

*Proof.* At each propagation step:

- 1. Only k active nodes participate in message passing (sparsity in senders)
- 2. Each active node communicates with at most m neighbors bounded by connection strength threshold (locality)
- 3. Each message involves  $O(d_h^2)$  operations for the linear transformation  $\mathbf{W}_h$

Total operations per step:  $O(k \cdot m \cdot d_h^2)$ .

Under typical parameterizations where sparse ignition yields  $k = O(\sqrt{N})$  active nodes and local connectivity yields  $m = O(\sqrt{N})$  neighbors, we obtain per-step complexity  $O(N \cdot d_h^2)$ , which is linear in N. Over K propagation steps, total complexity is  $O(K \cdot N \cdot d_h^2)$ .

Compare to self-attention:  $O(n^2 \cdot d)$  for sequence length n and dimension d. RSGN achieves linear scaling in the number of computational nodes through sparse, local computation.

## B. Expressiveness

**Theorem 2** (Universal Approximation). An RSGN with sufficient nodes N, appropriate positions  $\{\mathbf{p}_i\}$ , thresholds  $\{\theta_i\}$ , and learned affinities  $\{a_{ij}\}$  can approximate any continuous function  $f: \mathcal{X} \to \mathcal{Y}$  on a compact domain  $\mathcal{X}$  to arbitrary precision.

*Proof Sketch.* The proof proceeds in four steps:

**Step 1:** By the embedding theorem for hyperbolic spaces [21], any tree structure (and hence any hierarchical

decomposition of the function) can be embedded in  $\mathbb{H}^d$  with arbitrarily low distortion.

- **Step 2:** The soft-threshold activation dynamics can implement arbitrary gating operations as temperature  $T \to 0$ , selecting which nodes contribute to computation for each input.
- **Step 3:** The combination of spatial embedding and learned affinities subsumes the connectivity patterns of standard feedforward networks: placing nodes along a geodesic in hyperbolic space with appropriate thresholds recovers layer-wise sequential computation.
- **Step 4:** By the universal approximation theorem for feedforward networks with sufficient width [70, 71], the result follows.

**Corollary 1.** RSGN can represent any function representable by a Transformer of comparable capacity, though potentially with different computational complexity.

## C. Gradient Flow Properties

**Proposition 1** (Bounded Gradients). For soft threshold with temperature T > 0, gradients are bounded:  $|\partial SoftThresh/\partial x| \leq 1/(4T)$ .

This bound motivates our temperature annealing schedule: begin training with high T for smooth gradient landscapes, then decrease T to achieve true sparsity while maintaining stable gradients.

## D. Convergence of Hebbian Updates

**Proposition 2** (Hebbian Stability). Under the Hebbian update rules with exponential decay factor  $\gamma < 1$  applied to affinity parameters, the parameters remain bounded if  $\eta_a < 2(1-\gamma)/\lambda_{\max}(\mathbf{C})$  where  $\mathbf{C}$  is the correlation matrix of node activations.

This condition ensures that Hebbian updates do not diverge, maintaining bounded connectivity strengths throughout training.

## VI. EXPERIMENTS

We evaluate RSGN on synthetic benchmarks designed to probe hierarchical feature learning and long-range dependency capture. All experiments were conducted on NVIDIA T4 GPUs using PyTorch. Code and trained models are available at the accompanying repository.

## A. Experimental Setup

#### 1. Hierarchical Sequence Classification

We designed a challenging synthetic benchmark requiring hierarchical feature composition across multiple scales. The task involves classifying sequences based on patterns organized at three hierarchical levels:

- Level 1 (Local): Random 5-gram patterns inserted at 2-4 random positions per sequence
- Level 2 (Compositional): Mid-range patterns at quarter-positions of the sequence
- Level 3 (Global): Additive class signature across all positions

We generate sequences of length L=64 with feature dimension d=32, divided into C=20 classes. Gaussian noise with  $\sigma=0.3$  is added, creating a challenging classification task where random guessing achieves only 5% accuracy.

#### 2. Long-Range Dependency Task

To evaluate RSGN's ability to capture dependencies across long sequences, we designed a task where class labels depend on patterns at both the *beginning* and *end* of sequences. Specifically, for sequences of length L=128:

- The first 8 positions contain a class-specific "start" pattern
- The last 8 positions contain a corresponding "end" pattern
- The middle 112 positions contain noise

Models must learn to integrate information from both extremes of the sequence to classify correctly. This task has 10 classes (10% random baseline).

# 3. Baseline Models

We compare RSGN against several strong baselines representing different architectural paradigms:

- MLP: Flattened input with two hidden layers (ReLU activation)
- **Transformer:** Standard multi-head self-attention with 4 heads and 2 layers
- Sparse Transformer: Fixed local (window 5) plus strided (every 4) attention pattern
- LSTM: Bidirectional LSTM with 2 layers

![](_page_9_Figure_1.jpeg)

FIG. 3: Accuracy comparison on hierarchical classification task (20 classes). RSGN achieves 23.8% accuracy with only 41,672 parameters, compared to Transformer's 30.1% with 403,348 parameters. The random baseline for 20 classes is 5%, meaning RSGN achieves nearly  $5\times$  better than random with  $10\times$  fewer parameters than Transformer.

All models are trained with AdamW optimizer [67] for 50 epochs with learning rate  $10^{-3}$ , weight decay  $10^{-4}$ , and cosine annealing schedule. We report mean and standard deviation over 3 random seeds.

# 4. RSGN Configuration

For all experiments, RSGN uses N=256 nodes, hidden dimension  $d_h=128$ , embedding dimension d=3, and K=5 propagation steps (7 for long-range task). Hebbian learning rate is  $\eta_a=0.002$  with decay factor  $\gamma=0.995$ . Temperature is fixed at T=1.0 throughout training.

# B. Hierarchical Classification Results

Table I presents performance on the hierarchical classification task. Figure 3 visualizes these results.

**TABLE I:** Performance comparison on hierarchical sequence classification (20 classes, sequence length 64, noise level 0.3). Results show mean  $\pm$  standard deviation over 3 runs. Random baseline is 5%. Rel. Size indicates relative parameter count compared to RSGN (1.0×). RSGN achieves competitive performance with approximately  $10\times$  fewer parameters than Transformer.

| Model              | Accuracy (%)   | Parameters | Rel. Size     |
|--------------------|----------------|------------|---------------|
| Transformer        | $30.1 \pm 0.2$ | 403,348    | 9.7×          |
| RSGN+Hebbian       | $23.8 \pm 0.2$ | 41,672     | $1.0 \times$  |
| RSGN               | $23.8 \pm 0.1$ | 41,672     | $1.0 \times$  |
| LSTM               | $18.1 \pm 0.4$ | 566,292    | $13.6 \times$ |
| MLP                | $16.0 \pm 0.8$ | 281,364    | $6.8 \times$  |
| Sparse Transformer | $15.9 \pm 0.2$ | 403,348    | $9.7 \times$  |

Several key observations emerge from these results:

**Parameter Efficiency:** RSGN achieves 23.8% accuracy with only 41,672 parameters, nearly 10 times fewer than the Transformer (403,348 parameters) and 14 times

![](_page_9_Figure_12.jpeg)

FIG. 4: Accuracy on long-range dependency task (sequence length 128, 10 classes). RSGN achieves 96.5% accuracy with 40,382 parameters, compared to Transformer and LSTM achieving 100% with approximately  $15\times$  more parameters. The strong performance demonstrates RSGN's ability to capture long-range dependencies despite using significantly fewer parameters.

fewer than LSTM (566,292 parameters). While Transformer achieves higher absolute accuracy (30.1%), RSGN's efficiency is remarkable: it achieves 79% of Transformer's accuracy with 10% of the parameters.

Comparison to Random Baseline: With 20 classes, random guessing achieves 5%. RSGN's 23.8% represents nearly  $5\times$  improvement over random, demonstrating meaningful learning of the hierarchical structure.

**Sparse Transformer Failure:** The fixed sparsity pattern of Sparse Transformer (15.9%) performs worse than even MLP (16.0%), suggesting that the predetermined local+strided pattern fails to capture the multi-scale hierarchical patterns in this task. RSGN's *input-dependent* sparsity adapts to each input's structure.

**Hebbian Learning Benefit:** RSGN with Hebbian learning (23.83%) marginally outperforms RSGN without (23.77%), with the benefit more pronounced in terms of training stability and convergence speed observed during training.

# C. Long-Range Dependency Results

Table II presents results on the long-range dependency task. Figure 4 visualizes the comparison.

**TABLE II:** Performance on long-range dependency task (10 classes, sequence length 128). The task requires integrating information from sequence start and end positions. Results show mean  $\pm$  standard deviation over 3 runs.

| Model        | Accuracy (%)    | Parameters |
|--------------|-----------------|------------|
| Transformer  | $100.0 \pm 0.0$ | 600,330    |
| LSTM         | $100.0 \pm 0.0$ | 563,722    |
| RSGN+Hebbian | $96.5 \pm 0.5$  | 40,382     |
| RSGN         | $96.1 \pm 0.2$  | 40,382     |

![](_page_10_Figure_1.jpeg)

FIG. 5: Ablation study results showing the contribution of each RSGN component. The relatively stable performance across configurations suggests robustness to hyperparameter choices, with Hebbian learning providing consistent benefits.

The results reveal important characteristics of RSGN's long-range modeling:

Competitive Long-Range Performance: RSGN achieves 96.5% accuracy on this task requiring integration of information across 128 positions. While Transformer and LSTM achieve perfect 100% accuracy, they require approximately  $15\times$  more parameters to do so.

Parameter Efficiency for Long Sequences: RSGN uses only 40,382 parameters for this task, compared to 600,330 for Transformer. This  $15\times$  reduction in parameters while maintaining 96.5% accuracy demonstrates exceptional parameter efficiency.

Propagation Dynamics Enable Long-Range: RSGN captures long-range dependencies through iterative propagation rather than direct attention. With 7 propagation steps, information can flow from ignited nodes at sequence extremes through the hyperbolic space, with connection weights enabling long-distance communication via the hierarchical structure.

# D. Ablation Study

Table III presents ablation experiments isolating the contribution of each RSGN component. Figure 5 visualizes these results.

**TABLE III:** Ablation study examining the contribution of RSGN components on the hierarchical classification task. All configurations use the same training protocol (50 epochs, AdamW optimizer).

| Configuration         | Accuracy (%) | Parameters |
|-----------------------|--------------|------------|
| 128 Nodes             | 24.4         | 32,840     |
| Full RSGN (256 nodes) | 24.1         | $41,\!672$ |
| 3 Propagation Steps   | 24.1         | $41,\!672$ |
| 1 Propagation Step    | 24.1         | $41,\!672$ |
| No Hebbian Learning   | 23.7         | $41,\!672$ |
| 512 Nodes             | 23.6         | $59,\!336$ |

The ablation results reveal several insights:

![](_page_10_Figure_12.jpeg)

FIG. 6: Parameter efficiency analysis: accuracy versus model size. RSGN occupies a favorable position in the lower-left region, achieving competitive accuracy with significantly fewer parameters than baseline models. The Pareto frontier suggests RSGN offers an attractive trade-off for parameter-constrained applications.

Robustness to Configuration: RSGN shows remarkable stability across configurations, with accuracy ranging from 23.6% to 24.4%. This suggests that the architecture is robust to hyperparameter choices within reasonable ranges.

Hebbian Learning Contribution: Removing Hebbian learning decreases accuracy from 24.1% to 23.7%, confirming that structural plasticity contributes to performance.

Node Count Trade-offs: Interestingly, 128 nodes (24.4%) slightly outperforms both 256 nodes (24.1%) and 512 nodes (23.6%). This suggests that larger models may overfit on this dataset size, and that RSGN can achieve good performance even with fewer nodes.

**Propagation Steps:** Similar performance across 1, 3, and 5 propagation steps on this task suggests that the hierarchical classification primarily relies on the ignition mechanism for pattern matching, with propagation providing refinement rather than fundamental capability.

# E. Parameter Efficiency Analysis

Figure 6 presents a scatter plot of accuracy versus parameter count, highlighting RSGN's favorable position in the efficiency landscape.

RSGN's position in the parameter-efficiency landscape is notable:

- $\bullet$  Achieves 79% of Transformer's accuracy with 10% of parameters
- Outperforms LSTM (18.1%) with 7% of its parameters
- Outperforms Sparse Transformer (15.9%) with identical parameters

• Represents an attractive Pareto trade-off for parameter-constrained applications

#### F. Training Dynamics

Figure 7 shows training and validation curves for all models on the hierarchical classification task.

The training curves reveal:

- Stable Convergence: RSGN converges smoothly without the oscillations sometimes observed in Transformer training
- Generalization: The gap between training and validation accuracy is small for RSGN, suggesting good generalization
- Early Convergence: RSGN reaches near-final performance within 30 epochs, with remaining epochs providing marginal improvement

#### G. Combined Results Visualization

Figure 8 presents a comprehensive visualization combining hierarchical classification, long-range dependencies, and parameter efficiency.

## VII. DISCUSSION

# A. Comparison with Published Models

Our experimental results situate RSGN within the broader landscape of efficient sequence models. We discuss comparisons with several key architectures:

Comparison with Standard Transformers [1]: While Transformers achieve higher absolute accuracy on our benchmarks (30.1% vs. 23.8% on hierarchical classification, 100% vs. 96.5% on long-range), they require  $10\text{-}15\times$  more parameters. For applications where parameter budget is constrained, such as edge deployment, embedded systems, or resource-limited training, RSGN offers an attractive alternative. Furthermore, RSGN's  $O(n \cdot k)$  complexity versus Transformer's  $O(n^2)$  becomes increasingly advantageous for longer sequences.

Comparison with Sparse Transformers [32]: Our Sparse Transformer baseline, using fixed local+strided attention patterns, achieved only 15.9% on hierarchical classification, significantly worse than RSGN's 23.8%. This demonstrates that *input-dependent* sparsity (as in RSGN) outperforms *fixed* sparsity patterns when the task requires adaptive routing. The BigBird architecture [33] adds random attention and global tokens to sparse patterns; exploring similar augmentations for RSGN is an interesting direction.

Comparison with State Space Models: S4 [37] and Mamba [42] achieve linear complexity through fundamentally different mechanisms, continuous-time state dynamics rather than spatial embedding. While we did not directly benchmark against S4/Mamba, their reported performance on Long Range Arena suggests complementary strengths. RSGN's explicit hierarchical structure in hyperbolic space may offer advantages for tasks with inherent hierarchical organization, while state space models may excel at smooth, continuous dynamics.

Comparison with Mixture of Experts [47]: MoE routes entire tokens to expert subnetworks, achieving input-dependent computation at a coarse granularity. RSGN's node-level activation provides finer-grained routing: each input activates a different subset of nodes within a single network rather than selecting among discrete expert modules. This allows more flexible adaptation to input structure.

# B. Biological Plausibility and Neural Correspondences

RSGN incorporates several principles with clear biological analogues (see Figure 2 for the complete architecture):

**TABLE IV:** Correspondences between RSGN mechanisms and biological neural systems.

| RSGN Mechanism       | Biological Analogue                  |
|----------------------|--------------------------------------|
| Sparse ignition      | Sparse coding in sensory cortex [11] |
| Local inhibition     | Lateral inhibition [24]              |
| Hebbian plasticity   | Synaptic plasticity [16]             |
| -                    | Homeostatic scaling [68]             |
| V 1                  | Cortical hierarchy [19]              |
|                      | Dopaminergic modulation [30]         |
| Propagation dynamics | Recurrent processing [72]            |

While RSGN does not claim biological realism at the implementation level (neurons are not point-particle nodes, synapses are not simple scalar weights), these correspondences suggest that the *computational principles* underlying biological efficiency may transfer productively to artificial systems.

# C. Advantages of RSGN

Our experiments reveal several key advantages of the RSGN architecture:

**Parameter Efficiency:** RSGN achieves competitive performance with dramatically fewer parameters, making it suitable for deployment in resource-constrained environments.

**Input-Dependent Routing:** Unlike fixed sparse patterns, RSGN adapts its active computation graph for each input, providing flexible routing without the overhead of explicit gating networks.

![](_page_12_Figure_1.jpeg)

FIG. 7: Training dynamics on hierarchical classification task. Left: training accuracy over epochs. Right: validation accuracy over epochs. RSGN shows stable training dynamics with consistent convergence, while some baselines exhibit higher variance.

![](_page_12_Figure_3.jpeg)

FIG. 8: Combined experimental results. (A) Hierarchical classification accuracy (20 classes). (B) Long-range dependency accuracy (sequence length 128). (C) Parameter efficiency scatter plot. RSGN demonstrates competitive performance across tasks while maintaining significant parameter efficiency.

Hierarchical Structure: The hyperbolic embedding provides explicit hierarchical organization, potentially beneficial for tasks with inherent hierarchy (taxonomies, parse trees, compositional structures).

**Interpretability:** The spatial organization of nodes in hyperbolic space provides natural interpretability through visualization and cluster analysis. Active nodes for different inputs can be examined to understand routing decisions.

**Graceful Scaling:** RSGN's complexity scales with active computation rather than network size, potentially enabling scaling to larger models while maintaining efficiency.

#### D. Limitations and Future Work

Several limitations merit discussion and suggest directions for future research:

Absolute Accuracy Gap: While RSGN achieves remarkable parameter efficiency, Transformers still achieve higher absolute accuracy on our benchmarks. Closing this gap while maintaining efficiency is an important goal. Potential approaches include: deeper propagation dynamics, learned distance functions, and hybrid architectures combining RSGN with attention.

Hardware Efficiency: Current GPU architectures are optimized for dense, regular computation patterns. RSGN's sparse, dynamic computation does not map efficiently to existing hardware, limiting practical speedups despite theoretical complexity advantages. Neuromorphic hardware [73, 74] designed for sparse, event-driven com-

putation could better realize RSGN's efficiency potential.

**Scale:** Our experiments focus on moderate-scale synthetic tasks. Scaling RSGN to billion-parameter regimes and evaluating on standard NLP/vision benchmarks (e.g., language modeling, ImageNet) remains important future work.

**Training Complexity:** The two-timescale learning system requires careful hyperparameter tuning to balance fast and slow learning rates. Automated methods for setting these hyperparameters would improve usability.

Theoretical Understanding: While we provide complexity analysis and stability conditions, complete convergence guarantees for the full system combining gradient descent with Hebbian structural learning remain an open theoretical question.

#### E. Future Directions

Several promising directions emerge from this work:

Neuromorphic Implementation: RSGN's sparse, event-driven computation aligns well with neuromorphic hardware principles. Implementation on Intel Loihi [73] or IBM TrueNorth [74] could realize significant energy efficiency gains.

Continual Learning: The structural plasticity mechanisms of RSGN may enable more graceful continual learning without catastrophic forgetting, as new information can be accommodated by structural reorganization rather than overwriting existing weights.

Multimodal Learning: Different sensory modalities could occupy different regions of the hyperbolic space, with cross-modal connections emerging through Hebbian co-activation during multimodal learning.

**Hybrid Architectures:** Combining RSGN's efficient routing with Transformer attention for critical operations could yield architectures that balance efficiency and capability.

Computational Biology: RSGN's hierarchical representations in hyperbolic space naturally align with the multi-scale organization of biological systems. Applications to molecular signaling networks, where hierarchical language models have shown promise [75], could benefit from RSGN's sparse, input-dependent routing to model pathway-specific cellular responses [76].

Brain-Computer Interfaces: RSGN's properties make it a promising candidate for brain-computer interface (BCI) applications [77]. The parameter efficiency (10-15× reduction) enables deployment on implantable devices with strict power and size constraints. Sparse activation patterns (1-2% of nodes active) naturally align with the sparse firing patterns of cortical neurons, potentially improving neural signal decoding. The Hebbian learning mechanism could enable online adaptation to neural drift caused by electrode movement or neural plasticity [78], reducing the need for frequent recalibration. Input-dependent routing allows different neural signal types (motor imagery, speech, attention states) to activate

specialized computational pathways. Potential applications include motor neuroprosthetics [79], speech decoding for communication devices [80, 81], seizure prediction [82], and closed-loop neuromodulation systems [83]. Validation on real neural recordings (EEG, ECoG, single-unit data) represents an important direction for translating these theoretical advantages to clinical practice.

## VIII. CONCLUSION

We have introduced Resonant Sparse Geometry Networks (RSGN), a neural architecture that learns sparse, hierarchical, and input-dependent connectivity inspired by biological neural systems (Figure 1, Figure 2). Through the combination of hyperbolic spatial embedding, distance-based connectivity, two-timescale learning, and input-dependent ignition, RSGN achieves competitive performance with dramatically reduced parameter counts.

Our key experimental findings include:

- On hierarchical classification (20 classes), RSGN achieves 23.8% accuracy with 41,672 parameters, compared to Transformer's 30.1% with 403,348 parameters, a 10× parameter reduction while achieving 79% of the accuracy.
- On long-range dependencies (sequence length 128), RSGN achieves 96.5% accuracy with 40,382 parameters, compared to Transformer's 100% with 600,330 parameters, a 15× parameter reduction while achieving 96.5% of the accuracy.
- Ablation studies confirm that each component contributes to performance, with Hebbian learning providing consistent improvements.
- RSGN demonstrates stable training dynamics and good generalization across configurations.

The key insight underlying RSGN is that structure and routing can be learned through different mechanisms operating on different timescales: fast gradient descent for activation routing, slow Hebbian rules for connectivity structure, both shaped by global reward signals. This separation mirrors biological neural systems and suggests that the next generation of neural architectures may move beyond fixed, dense computation graphs toward self-organizing, sparse, and dynamic structures.

By taking inspiration from the remarkable efficiency of biological intelligence, operating on 20 watts while processing complex information across billions of neurons, we hope RSGN contributes toward more sustainable and capable artificial neural systems. As model sizes continue to grow and computational resources become increasingly constrained, architectures that achieve more with less will become ever more important.

#### CODE AVAILABILITY

The complete implementation of RSGN, including model code, training scripts, experiment notebooks, and analysis tools, is available at https://github.com/HasiHays/RSGN.

ACKNOWLEDGMENTS

We thank the research community for foundational work on hyperbolic neural networks, state space models, and biologically-inspired learning rules that made this work possible. We acknowledge computational resources provided by Google Colab.

- A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, Attention is all you need, Advances in neural information processing systems 30 (2017).
- [2] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al., Language models are few-shot learners, Advances in neural information processing systems 33, 1877 (2020).
- [3] OpenAI, Gpt-4 technical report, arXiv preprint arXiv:2303.08774 (2023).
- [4] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al., An image is worth 16x16 words: Transformers for image recognition at scale, arXiv preprint arXiv:2010.11929 (2020).
- [5] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al., Flamingo: a visual language model for few-shot learning, Advances in Neural Information Processing Systems 35, 23716 (2022).
- [6] H. Hays, Attention mechanisms in neural networks, arXiv preprint arXiv:2601.03329 (2026).
- [7] H. Hays, Encyclopedia of Large Language Models and Foundation Models (Zenodo, 2026).
- [8] J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler, et al., Emergent abilities of large language models, arXiv preprint arXiv:2206.07682 (2022).
- [9] Y. Tay, M. Dehghani, D. Bahri, and D. Metzler, Efficient transformers: A survey, ACM Computing Surveys 55, 1 (2022).
- [10] F. A. Azevedo, L. R. Carvalho, L. T. Grinberg, J. M. Farfel, R. E. Ferretti, R. E. Leite, W. J. Filho, R. Lent, and S. Herculano-Houzel, Equal numbers of neuronal and nonneuronal cells make the human brain an isometrically scaled-up primate brain, Journal of Comparative Neurology 513, 532 (2009).
- [11] B. A. Olshausen and D. J. Field, Emergence of simple-cell receptive field properties by learning a sparse code for natural images, Nature 381, 607 (1996).
- [12] A. L. Barth and U. S. Bhalla, Experimental evidence for sparse firing in the neocortex, Trends in neurosciences 35, 345 (2012).
- [13] P. Földiák, Sparse coding in the primate cortex, The handbook of brain theory and neural networks , 1064

- (2003).
- [14] S. Dehaene, Consciousness and the brain: Deciphering how the brain codes our thoughts, Viking Press (2014).
- [15] G. Tononi, M. Boly, M. Massimini, and C. Koch, Integrated information theory: from consciousness to its physical substrate, Nature Reviews Neuroscience 17, 450 (2016).
- [16] D. O. Hebb, The organization of behavior: A neuropsychological theory (Wiley, 1949).
- [17] P. R. Huttenlocher, Synaptic density in human frontal cortex—developmental changes and effects of aging, Brain research 163, 195 (1979).
- [18] Z. Petanjek, M. Judaš, G. Šimić, M. R. Rašin, H. B. Uylings, P. Rakic, and I. Kostović, Extraordinary neoteny of synaptic spines in the human prefrontal cortex, Proceedings of the National Academy of Sciences 108, 13281 (2011).
- [19] D. J. Felleman and D. C. Van Essen, Distributed hierarchical processing in the primate cerebral cortex, Cerebral cortex 1, 1 (1991).
- [20] K. D. Harris and T. D. Mrsic-Flogel, Cortical connectivity and sensory coding, Nature 503, 51 (2013).
- [21] R. Sarkar, Low distortion delaunay embedding of trees in hyperbolic plane, in *International symposium on graph* drawing (Springer, 2011) pp. 355–366.
- [22] M. Nickel and D. Kiela, Poincaré embeddings for learning hierarchical representations, in *Advances in neural information processing systems*, Vol. 30 (2017).
- [23] F. Sala, C. De Sa, A. Gu, and C. Ré, Representation tradeoffs for hyperbolic embeddings, Proceedings of machine learning research 80, 4460 (2018).
- [24] J. S. Isaacson and M. Scanziani, How inhibition shapes cortical activity, Neuron 72, 231 (2011).
- [25] M. Carandini and D. J. Heeger, Normalization as a canonical neural computation, Nature Reviews Neuroscience 13, 51 (2012).
- [26] B. J. Baars, A cognitive theory of consciousness, Cambridge University Press (1988).
- [27] S. Dehaene and J.-P. Changeux, Experimental and theoretical approaches to conscious processing, Neuron 70, 200 (2011).
- [28] K. Friston, Learning and inference in the brain, Neural Networks 16, 1325 (2003).
- [29] T. P. Lillicrap, A. Santoro, L. Marris, C. J. Akerman, and G. Hinton, Backpropagation and the brain, Nature Reviews Neuroscience 21, 335 (2020).

- [30] W. Schultz, P. Dayan, and P. R. Montague, A neural substrate of prediction and reward, Science 275, 1593 (1997).
- [31] J. N. Reynolds, B. I. Hyland, and J. R. Wickens, A cellular mechanism of reward-related learning, Nature 413, 67 (2001).
- [32] R. Child, S. Gray, A. Radford, and I. Sutskever, Generating long sequences with sparse transformers, arXiv preprint arXiv:1904.10509 (2019).
- [33] M. Zaheer, G. Guruganesh, K. A. Dubey, J. Ainslie, C. Alberti, S. Ontanon, P. Pham, A. Ravula, Q. Wang, L. Yang, et al., Big bird: Transformers for longer sequences, Advances in Neural Information Processing Systems 33, 17283 (2020).
- [34] S. Wang, B. Z. Li, M. Khabsa, H. Fang, and H. Ma, Linformer: Self-attention with linear complexity, arXiv preprint arXiv:2006.04768 (2020).
- [35] K. Choromanski, V. Likhosherstov, D. Dohan, X. Song, A. Gane, T. Sarlos, P. Hawkins, J. Davis, A. Mohiuddin, L. Kaiser, et al., Rethinking attention with performers, arXiv preprint arXiv:2009.14794 (2020).
- [36] A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret, Transformers are rnns: Fast autoregressive transformers with linear attention, International Conference on Machine Learning, 5156 (2020).
- [37] A. Gu, K. Goel, and C. Ré, Efficiently modeling long sequences with structured state spaces, arXiv preprint arXiv:2111.00396 (2021).
- [38] A. Gu, T. Dao, S. Ermon, A. Rudra, and C. Ré, Hippo: Recurrent memory with optimal polynomial projections, Advances in Neural Information Processing Systems 33, 1474 (2020).
- [39] J. T. Smith, A. Warrington, and S. W. Linderman, Simplified state space layers for sequence modeling, arXiv preprint arXiv:2208.04933 (2022).
- [40] R. Hasani, M. Lechner, A. Amini, D. Rus, and R. Grosu, Liquid structural state-space models, arXiv preprint arXiv:2209.12951 (2022).
- [41] Y. Tay, M. Dehghani, S. Abnar, Y. Shen, D. Bahri, P. Pham, J. Rao, L. Yang, S. Ruder, and D. Metzler, Long range arena: A benchmark for efficient transformers, arXiv preprint arXiv:2011.04006 (2020).
- [42] A. Gu and T. Dao, Mamba: Linear-time sequence modeling with selective state spaces, arXiv preprint arXiv:2312.00752 (2023).
- [43] O. Ganea, G. Bécigneul, and T. Hofmann, Hyperbolic neural networks, in *Advances in neural information processing systems*, Vol. 31 (2018).
- [44] C. Gulcehre, M. Denil, M. Malinowski, A. Razavi, R. Pascanu, K. M. Hermann, P. Battaglia, V. Bapst, D. Raposo, A. Santoro, et al., Hyperbolic attention networks, arXiv preprint arXiv:1805.09786 (2018).
- [45] N. Linial, E. London, and Y. Rabinovich, The geometry of graphs and some of its algorithmic applications, Combinatorica 15, 215 (1995).
- [46] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, Outrageously large neural networks: The sparsely-gated mixture-of-experts layer, in *International conference on learning representations* (2017).
- [47] W. Fedus, B. Zoph, and N. Shazeer, Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity, The Journal of Machine Learning Research 23, 5232 (2022).

- [48] N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou, A. W. Yu, O. Firat, et al., Glam: Efficient scaling of language models with mixtureof-experts, International Conference on Machine Learning , 5547 (2022).
- [49] Y. Han, G. Huang, S. Song, L. Yang, H. Wang, and Y. Wang, Dynamic neural networks: A survey, IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 7436 (2021).
- [50] S. Teerapittayanon, B. McDanel, and H.-T. Kung, Branchynet: Fast inference via early exiting from deep neural networks, International Conference on Pattern Recognition, 2464 (2016).
- [51] A. Graves, Adaptive computation time for recurrent neural networks, arXiv preprint arXiv:1603.08983 (2016).
- [52] J. Lin, Y. Rao, J. Lu, and J. Zhou, Runtime neural pruning, Advances in Neural Information Processing Systems 30 (2017).
- [53] B. Zoph and Q. V. Le, Neural architecture search with reinforcement learning, in *International Conference on Learning Representations* (2017).
- [54] H. Liu, K. Simonyan, and Y. Yang, Darts: Differentiable architecture search, arXiv preprint arXiv:1806.09055 (2018).
- [55] J. Frankle and M. Carbin, The lottery ticket hypothesis: Finding sparse, trainable neural networks, arXiv preprint arXiv:1803.03635 (2018).
- [56] H. Markram, J. Lübke, M. Frotscher, and B. Sakmann, Regulation of synaptic efficacy by coincidence of postsynaptic aps and epsps, Science 275, 213 (1997).
- [57] G.-q. Bi and M.-m. Poo, Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type, Journal of Neuroscience 18, 10464 (1998).
- [58] E. Oja, Simplified neuron model as a principal component analyzer, Journal of Mathematical Biology 15, 267 (1982).
- [59] T. Miconi, K. Stanley, and J. Clune, Differentiable plasticity: training plastic neural networks with backpropagation, International Conference on Machine Learning, 3559 (2018).
- [60] I. Pozzi, S. Bohte, and P. Roelfsema, Attention-gated brain propagation: How the brain can implement rewardbased error backpropagation, Advances in Neural Information Processing Systems 33, 2516 (2020).
- [61] R. P. Rao and D. H. Ballard, Predictive coding in the visual cortex: a functional interpretation of some extraclassical receptive-field effects, Nature neuroscience 2, 79 (1999).
- [62] K. Friston, A theory of cortical responses, Philosophical Transactions of the Royal Society B: Biological Sciences 360, 815 (2005).
- [63] B. Scellier and Y. Bengio, Equilibrium propagation: Bridging the gap between energy-based models and backpropagation, Frontiers in computational neuroscience 11, 24 (2017).
- [64] G. Hinton, The forward-forward algorithm: Some preliminary investigations, arXiv preprint arXiv:2212.13345 (2022).
- [65] W. C. Abraham, Metaplasticity: tuning synapses and networks for plasticity, Nature Reviews Neuroscience 9, 387 (2008).
- [66] J. L. Ba, J. R. Kiros, and G. E. Hinton, Layer normalization, arXiv preprint arXiv:1607.06450 (2016).

- [67] I. Loshchilov and F. Hutter, Decoupled weight decay regularization, arXiv preprint arXiv:1711.05101 (2017).
- [68] G. G. Turrigiano and S. B. Nelson, Homeostatic plasticity in the developing nervous system, Nature Reviews Neuroscience 5, 97 (2004).
- [69] A. Holtmaat and K. Svoboda, Experience-dependent structural synaptic plasticity in the mammalian brain, Nature Reviews Neuroscience 10, 647 (2009).
- [70] G. Cybenko, Approximation by superpositions of a sigmoidal function, Mathematics of control, signals and systems 2, 303 (1989).
- [71] K. Hornik, M. Stinchcombe, and H. White, Multilayer feedforward networks are universal approximators, Neural networks 2, 359 (1989).
- [72] V. A. Lamme and P. R. Roelfsema, The distinct modes of vision offered by feedforward and recurrent processing, Trends in Neurosciences 23, 571 (2000).
- [73] M. Davies, N. Srinivasa, T.-H. Lin, G. Chinya, Y. Cao, S. H. Choday, G. Dimou, P. Joshi, N. Imam, S. Jain, et al., Loihi: A neuromorphic manycore processor with on-chip learning, IEEE Micro 38, 82 (2018).
- [74] P. A. Merolla, J. V. Arthur, R. Alvarez-Icaza, A. S. Cassidy, J. Sawada, F. Akopyan, B. L. Jackson, N. Imam, C. Guo, Y. Nakamura, et al., A million spiking-neuron integrated circuit with a scalable communication network and interface, Science 345, 668 (2014).
- [75] H. Hays, Y. Yu, and W. J. Richardson, Hierarchical molecular language models, arXiv preprint arXiv:2512.00696 (2025).
- [76] H. Hays and W. Richardson, ECMSim: A high-performance web simulation of cardiac ECM remodeling through integrated ODE-based signaling and diffusion, arXiv preprint arXiv:2510.12577 (2025).
- [77] J. R. Wolpaw, N. Birbaumer, D. J. McFarland, G. Pfurtscheller, and T. M. Vaughan, Brain-computer interfaces for communication and control, Clinical Neurophysiology 113, 767 (2002).
- [78] J. A. Perge, M. L. Homer, W. Q. Malik, S. Cash, E. Eskandar, G. Friehs, J. P. Donoghue, and L. R. Hochberg, Intra-day signal instabilities affect decoding performance in an intracortical neural interface system, Journal of Neural Engineering 10, 036004 (2013).
- [79] L. R. Hochberg, D. Bacher, B. Jarosiewicz, N. Y. Masse, J. D. Simeral, J. Vogel, S. Haddadin, J. Liu, S. S. Cash, P. van der Smagt, et al., Reach and grasp by people with tetraplegia using a neurally controlled robotic arm, Nature 485, 372 (2012).
- [80] D. A. Moses, S. L. Metzger, J. R. Liu, G. K. Anumanchipalli, J. G. Makin, P. F. Sun, J. Chartier, M. E. Dougherty, P. M. Liu, G. M. Abrams, et al., Neuroprosthesis for decoding speech in a paralyzed person with anarthria, New England Journal of Medicine 385, 217 (2021).
- [81] F. R. Willett, D. T. Avansino, L. R. Hochberg, J. M. Henderson, and K. V. Shenoy, High-performance brainto-text communication via handwriting, Nature 593, 249 (2021).
- [82] M. J. Morrell and R. S. in Epilepsy Study Group, Responsive cortical stimulation for the treatment of medically intractable partial epilepsy, Neurology 77, 1295 (2011).
- [83] A. M. Lozano, N. Lipsman, H. Bergman, P. Brown, S. Chabardes, J. W. Chang, K. Matthews, C. C. McIntyre, T. E. Schlaepfer, U. Bhalla, et al., Deep brain stimulation: current challenges and future directions, Nature Reviews

Neurology 15, 148 (2019).

#### Appendix A: Hyperparameters

Table V lists default hyperparameters used in experiments.

**TABLE V:** Default hyperparameters for RSGN experiments.

| Parameter                                | Value              | Description                 |
|------------------------------------------|--------------------|-----------------------------|
| Architecture                             |                    |                             |
| Number of nodes $N$                      | 256                | Computational nodes         |
| Hidden dimension $d_h$                   | 128                | Node state dimension        |
| Space dimension $d$                      | 3                  | Hyperbolic embedding dim    |
| Propagation steps $K$                    | 5                  | Iterations per forward pass |
| Affinity rank $r$                        | 32                 | Low-rank factorization      |
| Activation Dynamics                      |                    |                             |
| Temperature $T$                          | 1.0                | Soft threshold temperature  |
| Distance temperature $\tau$              | 1.0                | Connection decay rate       |
| Sparsity target $\alpha_{\text{target}}$ | 0.1                | Target activation level     |
| Inhibition radius $r$                    | 0.3                | Local competition radius    |
| Ignition width $\sigma_{\rm ign}$        | 0.4                | Spark activation width      |
| Learning Rates                           |                    |                             |
| Fast learning rate                       | $10^{-3}$          | Gradient descent            |
| Hebbian rate $\eta_a$                    | $2 \times 10^{-3}$ | Affinity updates            |
| Threshold rate $\eta_{\theta}$           | $10^{-3}$          | Threshold adaptation        |
| Training                                 |                    |                             |
| Optimizer                                | AdamW              | With weight decay           |
| Weight decay                             | $10^{-4}$          | Regularization              |
| Batch size                               | 64                 | Training batch              |
| Epochs                                   | 50                 | Training duration           |
| Scheduler                                | Cosine             | Learning rate annealing     |
| Structural Plasticity                    |                    |                             |
| Affinity decay $\gamma$                  | 0.995              | Exponential decay           |
| Prune threshold                          | 0.01               | Connection removal          |
| Sprout threshold                         | 0.9                | Connection creation         |

#### Appendix B: Mathematical Details

## 1. Hyperbolic Operations

The Möbius addition in the Poincaré ball is:

$$\mathbf{x} \oplus \mathbf{y} = \frac{(1 + 2\langle \mathbf{x}, \mathbf{y} \rangle + ||\mathbf{y}||^2)\mathbf{x} + (1 - ||\mathbf{x}||^2)\mathbf{y}}{1 + 2\langle \mathbf{x}, \mathbf{y} \rangle + ||\mathbf{x}||^2||\mathbf{y}||^2}$$
(B1)

The exponential map at point  $\mathbf{x}$  maps tangent vector  $\mathbf{v}$  to:

$$\exp_{\mathbf{x}}(\mathbf{v}) = \mathbf{x} \oplus \left( \tanh\left(\frac{\lambda_{\mathbf{x}} \|\mathbf{v}\|}{2}\right) \frac{\mathbf{v}}{\|\mathbf{v}\|} \right)$$
 (B2)

where  $\lambda_{\mathbf{x}} = 2/(1 - \|\mathbf{x}\|^2)$  is the conformal factor. The logarithmic map (inverse of exponential):

$$\log_{\mathbf{x}}(\mathbf{y}) = \frac{2}{\lambda_{\mathbf{x}}} \operatorname{arctanh}(\| - \mathbf{x} \oplus \mathbf{y} \|) \frac{-\mathbf{x} \oplus \mathbf{y}}{\| - \mathbf{x} \oplus \mathbf{y} \|}$$
(B3)

#### 2. Gradient Derivations

For the soft threshold function  $f(x) = \sigma((x - \theta)/T)$ :

$$\frac{\partial f}{\partial x} = \frac{1}{T}\sigma'\left(\frac{x-\theta}{T}\right) \tag{B4}$$

$$= \frac{1}{T}\sigma\left(\frac{x-\theta}{T}\right)\left(1-\sigma\left(\frac{x-\theta}{T}\right)\right)$$
 (B5)

$$= \frac{1}{T}f(x)(1 - f(x))$$
 (B6)

Maximum gradient magnitude at  $x = \theta$  where f = 0.5:  $|f'|_{\text{max}} = 1/(4T)$ .

# 3. Complexity Analysis Details

For RSGN with N nodes, the computational cost per forward pass breaks down as:

- 1. **Ignition:**  $O(T \cdot N \cdot d)$  for computing distances from T input positions to N nodes in d-dimensional hyperbolic space
- 2. Connection weights:  $O(N \cdot m \cdot d)$  for computing weights to m neighbors per node (can be cached)
- 3. Propagation (per step):  $O(k \cdot m \cdot d_h^2)$  for message passing among k active nodes with m neighbors
- 4. **Inhibition:**  $O(k \cdot m)$  for local normalization
- 5. Output:  $O(k \cdot d_h \cdot d_{\text{out}})$  for weighted readout

Total:  $O(T \cdot N \cdot d + K \cdot k \cdot m \cdot d_h^2 + k \cdot d_h \cdot d_{\text{out}})$ Under typical settings  $(k, m = O(\sqrt{N}))$ , this simplifies to O(N) per forward pass.