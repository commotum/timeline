## 1. Basic Metadata
- Title: "Graph Perceiver IO: A General Architecture for Graph-Structured Data" (Title header)
- Authors: "Seyun Bae, Hoyoon Byun, Changdae Oh, Yoon-Sik Cho<sup>3</sup>, Kyungwoo Song<sup>4</sup>" (Highlights)
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary
The paper introduces "a Graph Perceiver IO (GPIO), the Perceiver IO for the graph structured dataset" to extend a general architecture to graph-structured data (Abstract).

## 3. Tasks Evaluated
| Task name | Task type | Dataset(s) | Domain | Evidence (quotes) |
| --- | --- | --- | --- | --- |
| Link prediction | Classification; Reasoning / relational | "Cora [49]"; "CiteSeer [50]"; "PubMed [51]" (Table 15, Section 8.4. Dataset) | "Cora, CiteSeer, and PubMed are all citation networks." (Section 8.4. Dataset) | "Besides the classification task, we validate our models on the link prediction task." (Section 5.1. Link Prediction) |
| Node classification | Classification | "Cora [49], CiteSeer [50], and PubMed [51]." (Section 5.2. Node Classification) | "Cora, CiteSeer, and PubMed are all citation networks." (Section 8.4. Dataset) | "For node classification experiments, we adopt the three benchmark data, Cora [49], CiteSeer [50], and PubMed [51]." (Section 5.2. Node Classification) |
| Graph classification | Classification | "MUTAG [52]"; "PROTEINS [52]"; "IMDB, REDDIT, and COLLAB [53]" (Section 5.3. Graph Classification) | "MUTAG is nitroaromatic compounds dataset aimed at predicting their mutagenicity on Salmonella typhimurium." "PROTEINS is a protein dataset classified as enzymes or non-enzymes, where nodes represent amino acids and edges connect those within a distance of less than 6 Angstroms." "IMDB-BINARY is a movie collaboration containing the ego-networks of 1,000 actors and actresses from IMDB." "REDDIT-BINARY is a dataset of Reddit discussion graphs, where nodes represent users and edges indicate replies." "COLLAB is a dataset of scientific collaboration where each graph represents a researcher's ego network, with nodes as the researcher and collaborators, and edges indicating collaborations." (Section 8.4. Dataset) | "We validate our models on the graph classification task." (Section 5.3. Graph Classification) |
| Multimodal few-shot image classification | Classification | "miniImageNet dataset." (Section 5.4. Multimodal Few-shot classification) | "We perform a few-shot image classification task to evaluate GPIO+ for image-graph multimodal learning." (Section 5.4. Multimodal Few-shot classification) | "Using two separated decoders, GPIO+ handle multimodal data such as image and graph." (Section 5.4. Multimodal Few-shot classification) |
| Multimodal text classification | Classification | "ogbn-products and ogbn-arxiv datasets [55]." (Table 5, Section 5.5. Multimodal Text Classification) | "We conduct evaluation on OGB datasets [55] which contains large graphs for text-graph multimodal learning." "The ogbn-products is a graph modeling Amazon's product co-purchasing network, which is undirected and unweighted." "The ogbn-arxiv dataset is a directed graph modeling citation relationships among Computer Science papers on ARXIV." (Section 5.5. Multimodal Text Classification; Section 8.4. Dataset) | "We conduct evaluation on OGB datasets [55] which contains large graphs for text-graph multimodal learning." (Section 5.5. Multimodal Text Classification) |

## 4. Domain and Modality Scope
Evaluation is performed on multiple domains within the graph modality and across multiple modalities: "We conduct comprehensive experiments, node classification, graph classification, link prediction, multimodal few-shot image classification, and multimodal text classification." (Section 5. Results) and "The GPIO is a general method that handles diverse datasets, such as graph-structured data, text, and images" (Abstract), with varied graph domains such as "MUTAG is nitroaromatic compounds dataset" and "IMDB-BINARY is a movie collaboration" (Section 8.4. Dataset). Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Link prediction | Not specified. | Not specified. | Task-specific output query array and inner-product decoder. | "For node-specific tasks such as node classification and link prediction, we set the output query array shape as M x D_q where M is the number of nodes." (Section 4.2. Output Query Array) "we adopt an inner product to predict edges." (Section 8.1.2. Learning for Link Prediction tasks) |
| Node classification | Not specified. | Not specified. | Task-specific output query array and logits layer. | "For node classification tasks, we set the size of the output query array to (M x D_q)." "The output array from the last cross attention layer is passed through the logits layer to make the vector size of each node equal to the class size." (Section 8.1.1. Learning for Node Classification tasks) |
| Graph classification | Not specified. | Not specified. | Task-specific output query array and logits layer. | "For graph classification task that requires a single label per graph, we set the output query array size as 1 x D_q." (Section 4.2. Output Query Array) |
| Multimodal few-shot image classification | Not specified. | Not specified. | Two decoders (task-specific heads). | "Using two separated decoders, GPIO+ handle multimodal data such as image and graph." (Section 5.4. Multimodal Few-shot classification) |
| Multimodal text classification | Not specified. | Not specified. | Not specified. | "The shape of the output query array depends on the given task." (Section 4.1. Overall Structure) |

## 6. Input and Representation Constraints
- Input size is variable and defined by graph size: "Perceiver IO handles various sizes of inputs and outputs" (Section 1. Introduction) and graph inputs are "X in R^{M x C} matrix" with adjacency "A in R^{M x M}" (Section 4.1. Overall Structure), so sequence length depends on M.
- Output query array size is fixed per task: "For node-specific tasks such as node classification and link prediction, we set the output query array shape as M x D_q" and "For graph classification task that requires a single label per graph, we set the output query array size as 1 x D_q" (Section 4.2. Output Query Array).
- Canonical positional information is injected by concatenating RWPE: "The methods are implemented by concatenating the node features and positional embeddings" (Section 4.1. Overall Structure) and "we concatenate the RWPE to obtain nodes representation" (Section 4.3. Input Array), with RWPE defined as "RWPE_i = [R_{ii}, R_{ii}^2, ..., R_{ii}^t] in R^t" (Section 4.3. Input Array).
- Fixed patch size is not used for images: "the Perceiver operates on the individual pixels independently without patches or convolution operations." (Section 3.2. Perceiver IO).
- Padding or resizing requirements: Not specified.

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified; input length is variable M as in "x in R^{M x C}" and "Perceiver IO handles various sizes of inputs and outputs" (Section 3.1. Perceiver; Section 1. Introduction).
- Attention type includes cross-attention and self-attention: "The Perceiver adopts the two types of attention, cross-attention and self-attention" (Section 3.1. Perceiver).
- Attention coverage is global (no windowing stated): "There is no restriction on the attention coverage, and it makes the GPIO handles the global structure information efficiently." (Section 4.3. Input Array).
- Computational cost management uses latent arrays: "They mitigate the quadratic scaling problem of self-attention blocks in Transformer by processing a small set of latent units instead of high-dimensional inputs." (Section 2.2. General architecture) and "The attention complexity of the Perceiver is O(NM) while the complexity of the self-attention for input-image is O(N^2)." (Section 3.1. Perceiver).

## 8. Positional Encoding (Critical Section)
- Base Perceiver PE options are learned or Fourier: "The Perceiver and the Perceiver IO introduce the learned positional encoding or Fourier-based positional encoding with sinusoid functions." (Section 3.2. Perceiver IO).
- Graph PE is RWPE: "we adopt the random walk positional embedding (RWPE) [41]." (Section 4.3. Input Array).
- PE is applied by concatenation at input: "The methods are implemented by concatenating the node features and positional embeddings" (Section 4.1. Overall Structure) and "we concatenate the RWPE to obtain nodes representation" (Section 4.3. Input Array).
- PE choices are compared in experiments: "Table 13 denotes how the presence and type of positional encoding affects graph classification performance." (Section 8.2.3. Graph Classification).

## 9. Positional Encoding as a Variable
- PE is treated as an experimental variable: "We analyze the impact of the smoothing times L and the effect of RWPE on link prediction." (Section 8.2.1. Link Prediction).
- Multiple PE types are compared: "+ None PE"; "+ Fourier PE"; "+ RWPE" (Table 13, Section 8.2.3. Graph Classification).
- The paper notes PE is not always critical: "In the Link prediction task, we can see that the presence of RWPE does not have a significant impact on performance, as enough relational information is captured by feature smoothing." (Section 8.2.1. Link Prediction).
- PE is a fixed architectural choice in the method description: "To reflect the topological information, specifically canonical positional information, with the original Perceiver input array structure, we adopt the random walk positional embedding (RWPE) [41]." (Section 4.3. Input Array).

## 10. Evidence of Constraint Masking
- Model size hyperparameters are explicitly bounded: "latent length | {16, 32, 64}" and "latent dimension | {32, 64}" (Table 6, Section 8.1.4. Hyperparameter Setting for Node Classification).
- Dataset sizes are fixed to specific benchmarks: "Cora [49] | 1 | 2,708 | 5,278 | 1,433 | 7" (Table 15, Section 8.4. Dataset) and "ogbn-arxiv | 169,343 | 1,166,243 | 40" (Table 16, Section 8.4. Dataset).
- Performance gains are attributed to architectural choices like smoothing: "We empirically find that the output query array with a smoothing method is one of the key factors for the GPIO." (Section 4.2. Output Query Array).
- Scaling model size or data as the primary driver of gains: Not claimed.

## 11. Architectural Workarounds
- Output query array smoothing injects relational information and is precomputed: "we propose the methods (i) output query array smoothing to incorporate relational information" (Section 4.1. Overall Structure) and "The smoothing operation is only once conducted, and it never appears during the model training." (Section 4.5. Complexity Analysis).
- RWPE concatenation adds canonical positional information: "we adopt the random walk positional embedding (RWPE) [41]." and "we concatenate the RWPE to obtain nodes representation" (Section 4.3. Input Array).
- Task-specific output query arrays replace read-out pooling: "For graph classification tasks, the output query array is 1-dimensional array." (Section 8.1.3. Learning for Graph Classification tasks) and "Then we pass the array to the logits layer (e.g., a single linear layer) instead of the read-out layer usually used for global pooling in graph-level tasks." (Section 8.3. Graph Related Tasks).
- GPIO+ uses two decoders for multimodal few-shot learning: "Using two separated decoders, GPIO+ handle multimodal data such as image and graph." (Section 5.4. Multimodal Few-shot classification).
- Latent-array attention reduces complexity: "They mitigate the quadratic scaling problem of self-attention blocks in Transformer by processing a small set of latent units instead of high-dimensional inputs." (Section 2.2. General architecture).

## 12. Explicit Limitations and Non-Claims
"However, our work does not scale well to domains where edge features are crucial, such as social graphs with multiple features like temporal dynamics and relationship types." (Section 7. Conclusion) "For further research, there is potential to extend to the study of canonical positional embeddings on domain specific graphs and various graph-related multimodal learning with output query design." and "It is necessary to propose a general architecture by adapting output query smoothing to incorporate edge features." (Section 7. Conclusion).

### 13. Constraint Profile (Synthesis)
> **Constraint Profile:**
> - Domain scope: Multiple graph domains plus multimodal image-graph and text-graph evaluations.
> - Task structure: Supervised classification and link prediction on fixed benchmarks; no open-ended tasks stated.
> - Representation rigidity: Inputs are node features concatenated with RWPE; output query array size is fixed per task.
> - Model sharing vs specialization: Single architecture is claimed, but task-specific output query arrays and a two-decoder GPIO+ indicate specialization; weight sharing across tasks is not specified.
> - Role of positional encoding: RWPE is central and compared against None/Fourier, and it is reported as not significant for link prediction.

### 14. Final Classification
Classification: Multi-task, multi-domain (constrained).

Justification: The evaluation spans multiple tasks and modalities, e.g., "node classification, graph classification, link prediction, multimodal few-shot image classification, and multimodal text classification" (Section 5. Results) and a model that "handles diverse datasets, such as graph-structured data, text, and images" (Abstract). At the same time, the experiments are bounded to fixed benchmark datasets with task-specific outputs ("The shape of the output query array depends on the given task." (Section 4.1. Overall Structure)), so the setup is constrained rather than unrestrained.
