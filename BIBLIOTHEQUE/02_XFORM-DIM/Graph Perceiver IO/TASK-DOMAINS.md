# Graph Perceiver IO: A General Architecture for Graph-Structured Data (Not specified in the paper.)
Source: Graph Perceiver IO.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| node classification | graph (node features + adjacency) | 1D (t); 2D (x, y) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | node labels/classes | 1D (t) (inferred) | Open (inferred) |
| graph classification | graph (node features + adjacency) | 1D (t); 2D (x, y) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | graph label/class | 0D (inferred) | Fixed (inferred) |
| link prediction | graph (node features + adjacency) | 1D (t); 2D (x, y) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | edge existence scores / predicted adjacency | 2D (x, y) (inferred) | Open (inferred) |
| multimodal few-shot image classification | images (support/query) as graph nodes with relations | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | class labels for query images | 1D (t) (inferred) | Capped (inferred) |
| multimodal text classification | text nodes in a graph (titles/abstracts/product descriptions + edges) | 1D (t); 2D (x, y) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | text node labels/classes | 1D (t) (inferred) | Open (inferred) |

## Summary
GPIO/GPIO+ are evaluated on graph-structured tasks (node classification, graph classification, and link prediction) plus multimodal text-graph and image-graph few-shot classification. Inputs are graphs with node features (including text or image features) and adjacency-derived structure, producing node labels, graph labels, edge predictions, or query-image labels. The justified dimensionality spans 0D, 1D, and 2D outputs with variable-size graph inputs and capped few-shot episodes; attention is global/static and state is constructed via latent arrays (inferred).

## Evidence
### Task: node classification
- "For node classification experiments, we adopt the three benchmark data, Cora [49], CiteSeer [50], and PubMed [51]." (Section 5.2. Node Classification)
- "For node classification tasks, we set the size of the output query array to  $(M \times D_q)$ ." (Section 8.1.1. Learning for Node Classification tasks)
- "The size of the output array is, then,  $(M \times E)$ ." (Section 8.1.1. Learning for Node Classification tasks)
- "Graph data usually consists of  $X \in \mathbb{R}^{M \times C}$  matrix, a set of nodes features, and adjacency matrix  $A \in \mathbb{R}^{M \times M}$ ," (Section 4.1. Overall Structure)
- Inference: Dimensions/dynamics inferred from the graph input/output size statements above and "Perceiver IO handles various sizes of inputs and outputs," (Section 1. Introduction). Attention/State inferred from "First, the cross-attention computes the attention between the latent arrays and the input arrays" and "The latent arrays  $z \in \mathbb{R}^{N \times D}$  are learnable latent representations," (Section 3.1. Perceiver) plus "There is no restriction on the attention coverage," (Section 4.3. Input Array)

### Task: graph classification
- "We validate our models on the graph classification task." (Section 5.3. Graph Classification)
- "For graph classification task that requires a single label per graph, we set the output query array size as  $1 \times D_q$ ," (Section 4.2. Output Query Array)
- "Graph data usually consists of  $X \in \mathbb{R}^{M \times C}$  matrix, a set of nodes features, and adjacency matrix  $A \in \mathbb{R}^{M \times M}$ ," (Section 4.1. Overall Structure)
- Inference: Dimensions/dynamics inferred from the graph input statement above and "Perceiver IO handles various sizes of inputs and outputs," (Section 1. Introduction). Attention/State inferred from "First, the cross-attention computes the attention between the latent arrays and the input arrays" and "The latent arrays  $z \in \mathbb{R}^{N \times D}$  are learnable latent representations," (Section 3.1. Perceiver) plus "There is no restriction on the attention coverage," (Section 4.3. Input Array)

### Task: link prediction
- "we validate our models on the link prediction task." (Section 5.1. Link Prediction)
- "we adopt an inner product to predict edges." (Section 8.1.2. Learning for Link Prediction tasks)
- "Matrix **A** denotes the adjacency matrix, and **W** is the output array." (Section 8.1.2. Learning for Link Prediction tasks)
- "Graph data usually consists of  $X \in \mathbb{R}^{M \times C}$  matrix, a set of nodes features, and adjacency matrix  $A \in \mathbb{R}^{M \times M}$ ," (Section 4.1. Overall Structure)
- Inference: Dimensions/dynamics inferred from the graph input statement above, the adjacency-matrix output description, and "Perceiver IO handles various sizes of inputs and outputs," (Section 1. Introduction). Attention/State inferred from "First, the cross-attention computes the attention between the latent arrays and the input arrays" and "The latent arrays  $z \in \mathbb{R}^{N \times D}$  are learnable latent representations," (Section 3.1. Perceiver) plus "There is no restriction on the attention coverage," (Section 4.3. Input Array)

### Task: multimodal few-shot image classification
- "We perform a few-shot image classification task to evaluate GPIO+ for image-graph multimodal learning." (Section 5.4. Multimodal Few-shot classification)
- "We configure input array as samples of Q and output query array as samples of Q and  $\mathcal{S}$  in the both decoders." (Section 4.6. Graph Perceiver IO+)
- "n-way k-shot classification problem means that support set S contains number of n classes" (Section 4.6. Graph Perceiver IO+)
- "The final class of query set selects one of the class of all support set" (Section 4.6. Graph Perceiver IO+)
- Inference: Dimensions/dynamics inferred from the few-shot set construction above and the n-way k-shot definition. Attention/State inferred from "First, the cross-attention computes the attention between the latent arrays and the input arrays" and "The latent arrays  $z \in \mathbb{R}^{N \times D}$  are learnable latent representations," (Section 3.1. Perceiver) plus "There is no restriction on the attention coverage," (Section 4.3. Input Array)

### Task: multimodal text classification
- "We conduct evaluation on OGB datasets [55] which contains large graphs for text-graph multimodal learning." (Section 5.5. Multimodal Text Classification)
- "The nodes of the graph consist of the title, summary, or product description of the article," (Section 5.5. Multimodal Text Classification)
- "The model should classify these text nodes into the correct class using topological information." (Section 5.5. Multimodal Text Classification)
- Inference: Dimensions/dynamics inferred from the graph input description above and "Perceiver IO handles various sizes of inputs and outputs," (Section 1. Introduction). Attention/State inferred from "First, the cross-attention computes the attention between the latent arrays and the input arrays" and "The latent arrays  $z \in \mathbb{R}^{N \times D}$  are learnable latent representations," (Section 3.1. Perceiver) plus "There is no restriction on the attention coverage," (Section 4.3. Input Array)
