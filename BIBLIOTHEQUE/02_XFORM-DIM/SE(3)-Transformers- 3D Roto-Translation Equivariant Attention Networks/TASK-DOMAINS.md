# SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks (Not specified in the paper.)
Source: SE(3)-Transformers- 3D Roto-Translation Equivariant Attention Networks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N-body dynamics prediction | Particle position, velocity, and charge at a time step | 3D (x, y, z) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | Relative particle location and velocity 500 time steps into the future | 3D (x, y, z) (inferred) | Fixed (inferred) |
| Object classification | 3D point cloud coordinates | 3D (x, y, z) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Object category label | 0D (inferred) | Fixed (inferred) |
| Molecular property regression | Molecular graph with atom one-hot node embeddings, bond types, and atom positions | 3D (x, y, z) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Molecular property values (6 reported regression targets) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates three task intents: N-body dynamics prediction, real-world object classification, and molecular property regression. Inputs are consistently 3D point/graph structures, while outputs span 3D future particle states and 0D labels/properties. Input dynamics are fixed for the N-body setup, capped for QM9 (up to 29 atoms), and not explicitly specified for ScanObjectNN in the OCR text. The architecture uses data-dependent self-attention and layered feature construction, supporting Dynamic attention and Constructed state as inferences from the method description.

## Evidence
### Task: N-body dynamics prediction
- "The N-body problem is an equivariant task: rotation of the input should result in rotated predictions of locations and velocities of the particles." (Section 4 Experiments)
- "The input to the network is the position of a particle in a specific time step, its velocity, and its charge. The task of the algorithm is then to predict the relative location and velocity 500 time steps into the future." (Section 4.1 N-Body Simulations)
- "Five particles each carry either a positive or a negative charge and exert repulsive or attractive forces on each other." (Section 4.1 N-Body Simulations)
- Inference: In/Out Dimension is 3D from particle positions/locations ("coordinate vectors  $\\mathbf{x}_i \\in \\mathbb{R}^3$" in Section 2 Background And Related Work). In/Out Dynamics is Fixed from the explicit five-particle setup (Section 4.1 N-Body Simulations). Attention Dynamic is inferred as Dynamic from data-dependent attention ("Attention weights" and "attention is performed" in Figure 1 caption and Section 3.2), and State Dynamic is inferred as Constructed from stacked equivariant layers plus self-interaction/pooling heads (Figure 1 caption; Section 3.2).

### Task: Object classification
- "Next, we evaluate on a real-world object classification task." (Section 4 Experiments)
- "To test our method, we choose ScanObjectNN, a recently introduced dataset for real-world object classification. The benchmark provides point clouds of 2902 objects across 15 different categories. We only use the coordinates of the points as input and object categories as training labels." (Section 4.2 Real-World Object Classification on ScanObjectNN)
- "For classification, this is followed by an invariant pooling layer and an MLP." (Figure 1 caption)
- Inference: Input Dimension is 3D from point-cloud coordinates (Section 4.2 and "coordinate vectors  $\\mathbf{x}_i \\in \\mathbb{R}^3$" in Section 2). Output Dimension is 0D for category labels, and Out Dynamics is Fixed for a single category decision per object (Section 4.2). Attention Dynamic is inferred as Dynamic from feature-dependent attention computations (Section 3.2), and State Dynamic is inferred as Constructed from multi-layer learned representations and pooling/MLP prediction head (Figure 1 caption; Section 3.2).

### Task: Molecular property regression
- "Finally, we test the SE(3)-Transformer on a molecular property regression task, which shines light on its ability to incorporate rich graph structures." (Section 4 Experiments)
- "The QM9 regression dataset [21] is a publicly available chemical property prediction task. There are 134k molecules with up to 29 atoms per molecule." (Section 4.3 OM9)
- "Atoms are represented as a 5 dimensional one-hot node embeddings in a molecular graph connected by 4 different chemical bond types" and "'Positions' of each atom are provided." (Section 4.3 OM9)
- "We show results on the test set of Anderson et al. [1] for 6 regression tasks in Table 3." (Section 4.3 OM9)
- Inference: Input Dimension is 3D from atom positions in a point/graph setting (Section 4.3 and Section 2). In Dynamics is Capped from "up to 29 atoms per molecule" (Section 4.3). Output Dimension is 0D for scalar property targets, and Out Dynamics is Fixed for the reported target set (Section 4.3/Table 3). Attention Dynamic is inferred as Dynamic from data-dependent attention (Section 3.2), and State Dynamic is inferred as Constructed from layered feature transformations and learned self-interaction (Section 3.2).
