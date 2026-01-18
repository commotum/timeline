## 1. Basic Metadata

- Title: "SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks" (Title/header)
- Authors: "Fabian B. Fuchs*†"; "Volker Fischer"; "Daniel E. Worrall*"; "Max Welling" (Author header)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

Abstract: "We introduce the SE(3)-Transformer, a variant of the self-attention module for 3D point clouds and graphs, which is *equivariant* under continuous 3D rototranslations."

## 3. Tasks Evaluated

### Task 1: N-Body simulations (future position/velocity prediction)

- Task name: "N-Body Simulations" (Section 4.1)
- Task type: Other (regression / dynamics prediction). Quote: "The task of the algorithm is then to predict the relative location and velocity 500 time steps into the future." (4.1 N-Body Simulations)
- Dataset(s) used: "an adaptation of the dataset from Kipf et al. [14]." (4.1 N-Body Simulations); "toy N-body particle simulation dataset" (Abstract)
- Domain: synthetic particle simulation. Quote: "Five particles each carry either a positive or a negative charge and exert repulsive or attractive forces on each other." (4.1 N-Body Simulations)

### Task 2: Real-world object classification on ScanObjectNN

- Task name: "Real-World Object Classification on ScanObjectNN" (Section 4.2)
- Task type: Classification. Quote: "real-world object classification task." (4 Experiments); "object categories as training labels." (4.2 Real-World Object Classification on ScanObjectNN)
- Dataset(s) used: "ScanObjectNN, a recently introduced dataset for real-world object classification. The benchmark provides point clouds of 2902 objects across 15 different categories." (4.2 Real-World Object Classification on ScanObjectNN)
- Domain: real-world 3D object point clouds. Quote: "point clouds of 2902 objects across 15 different categories." (4.2 Real-World Object Classification on ScanObjectNN)

### Task 3: Molecular property regression on QM9

- Task name: "molecular property regression task" (4 Experiments)
- Task type: Other (regression / property prediction). Quote: "The QM9 regression dataset [21] is a publicly available chemical property prediction task." (4.3 OM9)
- Dataset(s) used: "The QM9 regression dataset [21]" and "There are 134k molecules with up to 29 atoms per molecule." (4.3 OM9)
- Domain: molecular graphs / structures. Quote: "Atoms are represented as a 5 dimensional one-hot node embeddings in a molecular graph connected by 4 different chemical bond types" (4.3 OM9)

## 4. Domain and Modality Scope

- Evaluation scope: Multiple domains within the same modality (3D point clouds/graphs). Evidence: "We test the efficacy of the SE(3)-Transformer on three datasets, each testing different aspects of the model. The N-body problem is an equivariant task... Next, we evaluate on a real-world object classification task... Finally, we test the SE(3)-Transformer on a molecular property regression task" (4 Experiments); "a variant of the self-attention module for 3D point clouds and graphs" (Abstract).
- Single domain? No; multiple domains are explicitly evaluated (4 Experiments).
- Multiple modalities? Not indicated; all tasks are framed as "3D point clouds and graphs" (Abstract).
- Domain generalization / cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| N-Body simulations | Not stated. | Not stated. | Not stated. | "We trained an SE(3)-Transformer with 4 equivariant layers, each followed by an attentive self-interaction layer" (4.1 N-Body Simulations). |
| ScanObjectNN classification | Not stated. | Not stated. | Yes (max-pooling and MLP). | "We train an SE(3)-Transformer with 4 equivariant layers with linear self-interaction followed by max-pooling and an MLP." (4.2 Real-World Object Classification on ScanObjectNN) |
| QM9 regression | Not stated. | Not stated. | Not stated. | "We show results on the test set of Anderson et al. [1] for 6 regression tasks in Table 3." (4.3 OM9) |

## 6. Input and Representation Constraints

- Variable number of points: "operate on large point clouds and graphs with varying number of points" (Abstract).
- 3D coordinate inputs with optional features: "a point cloud as input, represented as a collection of n coordinate vectors  $\mathbf{x}_i \in \mathbb{R}^3$  with optional per-point features  $\mathbf{f}_i \in \mathbb{R}^d$ ." (2 Background And Related Work)
- Graph neighborhoods define local structure: "These neighbourhoods are computed either via the nearest-neighbours methods or may already be defined. For instance, molecular structures have neighbourhoods defined by their bonding structure." (3.1 Neighbourhoods)
- N-body input fields: "The input to the network is the position of a particle in a specific time step, its velocity, and its charge." (4.1 N-Body Simulations)
- ScanObjectNN inputs: "We only use the coordinates of the points as input and object categories as training labels." (4.2 Real-World Object Classification on ScanObjectNN)
- QM9 inputs: "Atoms are represented as a 5 dimensional one-hot node embeddings in a molecular graph connected by 4 different chemical bond types" (4.3 OM9)
- Fixed input resolution / patch size / token count / padding or resizing: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Sequence length fixed or variable: Variable; "varying number of points" (Abstract).
- Attention type: Sparse/local (neighborhood-based). Evidence: "Attention scales quadratically with point cloud size, so it is useful to introduce neighbourhoods: instead of each point attending to *all* other points, it only attends to its nearest neighbours." (2.2 Graph Neural Networks)
- Cost management: "Neighbourhoods reduce the computational complexity of the attention mechanism from quadratic in the number of points to linear." (3.1 Neighbourhoods); "Attention is performed on a per-neighbourhood basis" (3.2 The SE(3)-Transformer).

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: Implicit/relative use of coordinates; no explicit positional encoding is stated. Evidence: "\mathbf{W}_{V}^{\ell k}(\mathbf{x}_{j} - \mathbf{x}_{i})\mathbf{f}_{\text{in},j}^{k}" (Eq. 10, 3.2 The SE(3)-Transformer); "\mathbf{k}_{ij} = ... \mathbf{W}_K^{\ell k} (\mathbf{x}_j - \mathbf{x}_i) \mathbf{f}_{\mathrm{in},j}^k." (Eq. 11, 3.2 The SE(3)-Transformer); "keys, queries and values, which depend both on features and relative positions in a rotation-equivariant manner." (Figure 2 caption)
- Where applied: Relative positions appear in attention keys and value messages (Eq. 10 and Eq. 11, 3.2 The SE(3)-Transformer).
- Fixed across experiments / modified per task / ablated: Not specified.

## 9. Positional Encoding as a Variable

- Core research variable vs fixed assumption: Not specified.
- Multiple positional encodings compared: Not specified.
- Claims PE choice is not critical or secondary: Not specified.

## 10. Evidence of Constraint Masking

- Model size(s): "We trained an SE(3)-Transformer with 4 equivariant layers, each followed by an attentive self-interaction layer" (4.1 N-Body Simulations); "We train an SE(3)-Transformer with 4 equivariant layers with linear self-interaction followed by max-pooling and an MLP." (4.2 Real-World Object Classification on ScanObjectNN)
- Dataset size(s): "Five particles each carry either a positive or a negative charge" (4.1 N-Body Simulations); "The benchmark provides point clouds of 2902 objects across 15 different categories." (4.2 Real-World Object Classification on ScanObjectNN); "There are 134k molecules with up to 29 atoms per molecule." (4.3 OM9)
- Scaling model size: "This speed-up allowed us to train significantly larger versions of both the SE(3)-Transformer and the Tensor Field network [28] and to apply these models to real-world datasets." (5 Conclusion)
- Performance gains attribution: "Our experiments showed that adding attention to a roto-translation-equivariant model consistently led to higher accuracy and increased training stability. Specifically for large neighbourhoods, attention proved essential for model convergence." (5 Conclusion)
- Scaling data or training tricks as primary drivers: Not claimed.

## 11. Architectural Workarounds

- Neighborhood-based sparsification: "Attention scales quadratically with point cloud size, so it is useful to introduce neighbourhoods: instead of each point attending to *all* other points, it only attends to its nearest neighbours." (2.2 Graph Neural Networks); "Neighbourhoods reduce the computational complexity of the attention mechanism from quadratic in the number of points to linear." (3.1 Neighbourhoods)
- Graph construction: "The introduction of neighbourhoods converts our point cloud into a graph." (3.1 Neighbourhoods)
- Per-neighborhood attention: "Attention is performed on a per-neighbourhood basis" (3.2 The SE(3)-Transformer).
- Self-interaction skip connections: "Self-interaction is an elegant form of learnable skip connection" (3.2 The SE(3)-Transformer).
- Task-specific pooling head: "For classification, this is followed by an invariant pooling layer and an MLP." (Figure 1 caption); "followed by max-pooling and an MLP." (4.2 Real-World Object Classification on ScanObjectNN)
- Symmetry-adjusted variant: "We create an SO(2) invariant version of our algorithm by additionally feeding the z-component as an type-0 field and the x, y position as an additional type-1 field... We dub this model SE(3)-Transformer +z." (4.2 Real-World Object Classification on ScanObjectNN)

## 12. Explicit Limitations and Non-Claims

- Future work / limitation in N-body setup: "We deliberately formulated this as a regression problem to avoid the need to predict multiple time steps iteratively. Even though it certainly is an interesting direction for future research to combine equivariant attention with, e.g., an LSTM, our goal here was to test our core contribution and compare it to related models." (4.1 N-Body Simulations)
- Limitation due to symmetry assumptions in ScanObjectNN: "the task is not fully rotation invariant, in a statistical sense, as the objects are aligned with respect to the gravity axis. This results in a performance loss when deploying a fully SO(3) invariant model" (4.2 Real-World Object Classification on ScanObjectNN).
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not specified.

### 13. Constraint Profile (Synthesis)

**Constraint Profile:**
- Domain scope: Multiple domains (simulated N-body physics, real-world object scans, molecular graphs) within a single 3D point cloud/graph modality.
- Task structure: Separate, task-specific evaluations (dynamics regression, object classification, molecular property regression) rather than a unified multi-task training setup.
- Representation rigidity: Inputs are 3D point clouds/graphs with coordinates in R^3, optional node features, and neighborhood-defined graph structure; variable number of points.
- Model sharing vs specialization: Per-task training configurations are described, with no explicit shared-weight multi-task training across datasets.
- Role of positional encoding: No explicit positional encoding; relative positions (x_j - x_i) are used within attention computations.

### 14. Final Classification

**Classification:** Multi-task, multi-domain (constrained).

The paper evaluates three different tasks across distinct domains: "The N-body problem... real-world object classification task... molecular property regression task" (4 Experiments). At the same time, all evaluations are framed around "3D point clouds and graphs" (Abstract) and no cross-domain transfer is claimed, so the scope is multi-domain but constrained to a single 3D modality.
