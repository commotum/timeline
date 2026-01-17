## 1. Basic Metadata

- Title: "Do Transformers Really Perform Bad for Graph Representation?" (Quote, Title block: "# Do Transformers Really Perform Bad for Graph Representation?")
- Authors: "Chengxuan Ying*, Tianle Cai, Shengjie Luo*, Shuxin Zheng†, Guolin Ke, Di He†, Yanming Shen, Tie-Yan Liu" (Quote, Title block: "Chengxuan Ying<sup>1</sup>*, Tianle Cai<sup>2</sup>, Shengjie Luo<sup>3</sup>*, Shuxin Zheng<sup>4</sup>†, Guolin Ke<sup>4</sup>, Di He<sup>4</sup>†, Yanming Shen<sup>1</sup>, Tie-Yan Liu<sup>4</sup>")
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

"In this paper, we solve this mystery by presenting Graphormer, which is built upon the standard Transformer architecture, and could attain excellent results on a broad range of graph representation learning tasks." (Quote, Abstract)

## 3. Tasks Evaluated

- Task name: PCQM4M-LSC (quantum chemistry graph-level prediction)
  - Task type: Regression
  - Dataset(s) used: PCQM4M-LSC
  - Domain: Molecular graphs (2D molecular graphs)
  - Evidence: "PCQM4m-LSC is a quantum chemistry graph-level prediction task in recent OGB Large-Scale Challenge" (Quote, Section B.1 Details of Datasets); "The task of PCQM4M-LSC is to predict DFT(density functional theory)-calculated HOMO-LUMO energy gap of molecules given their 2D molecular graphs" (Quote, Section B.1 Details of Datasets); "PCQM4M-LSC     | Large  | 3,803,453 | 53,814,542 | 55,399,880 | Regression" (Quote, Table 6: Statistics of the datasets)

- Task name: OGBG-MolPCBA (molecular property prediction)
  - Task type: Classification
  - Dataset(s) used: OGBG-MolPCBA
  - Domain: Molecular graphs
  - Evidence: "we conduct experiments on two molecular graph datasets in popular OGB leaderboards, i.e., OGBG-MolPCBA and OGBG-MolHIV" (Quote, Section B.1 Details of Datasets); "OGBG-MolPCBA   | Medium | 437,929   | 11,386,154 | 12,305,805 | Binary classification" (Quote, Table 6: Statistics of the datasets)

- Task name: OGBG-MolHIV (molecular property prediction)
  - Task type: Classification
  - Dataset(s) used: OGBG-MolHIV
  - Domain: Molecular graphs
  - Evidence: "we conduct experiments on two molecular graph datasets in popular OGB leaderboards, i.e., OGBG-MolPCBA and OGBG-MolHIV" (Quote, Section B.1 Details of Datasets); "OGBG-MolHIV    | Small  | 41,127    | 1,048,738  | 1,130,993  | Binary classification" (Quote, Table 6: Statistics of the datasets)

- Task name: ZINC (graph property regression for solubility)
  - Task type: Regression
  - Dataset(s) used: ZINC (sub-set)
  - Domain: Molecular graphs (real-world molecular dataset)
  - Evidence: "We use the ZINC datasets, which is the most popular real-world molecular dataset to predict graph property regression for contrained solubility" (Quote, Section B.1 Details of Datasets); "ZINC (sub-set) | Small  | 12,000    | 277,920    | 597,960    | Regression" (Quote, Table 6: Statistics of the datasets)

## 4. Domain and Modality Scope

- Domain scope: Single domain (molecular graphs across datasets). Evidence: "The task of PCQM4M-LSC is to predict DFT(density functional theory)-calculated HOMO-LUMO energy gap of molecules given their 2D molecular graphs" (Quote, Section B.1 Details of Datasets); "we conduct experiments on two molecular graph datasets in popular OGB leaderboards, i.e., OGBG-MolPCBA and OGBG-MolHIV" (Quote, Section B.1 Details of Datasets); "We use the ZINC datasets, which is the most popular real-world molecular dataset" (Quote, Section B.1 Details of Datasets).
- Multiple domains within the same modality: Not stated; all datasets are described as molecular graphs (see quotes above).
- Multiple modalities: Not stated.
- Domain generalization or cross-domain transfer: Not claimed. The paper discusses within-domain transfer: "we mainly explore the transferable capability of a Graphormer model pre-trained on OGB-LSC (i.e., PCQM4M-LSC)" (Quote, Section 4.2 Graph Representation).

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| PCQM4M-LSC | Yes (serves as pretraining source for other tasks) | Not specified. | Not specified. | "we mainly explore the transferable capability of a Graphormer model pre-trained on OGB-LSC (i.e., PCQM4M-LSC)" (Quote, Section 4.2 Graph Representation); "We primarily report results on two model sizes: Graphormer (L=12, d=768), and a smaller one Graphormer<sub>SMALL</sub> (L=6, d=512)." (Quote, Section 4.1 OGB Large-Scale Challenge) |
| OGBG-MolPCBA | Yes | Yes | Not specified. | "we mainly explore the transferable capability of a Graphormer model pre-trained on OGB-LSC (i.e., PCQM4M-LSC)" (Quote, Section 4.2 Graph Representation); "Fine-tuning. Table 8 summarizes the hyper-parameters used for fine-tuning Graphormer on OGBG-MolPCBA." (Quote, Section B.2.2 OGBG-MolPCBA) |
| OGBG-MolHIV | Yes | Yes | Not specified. | "We use the Graphormer reported in Table 1 as the pre-trained model for OGBG-MolHIV" (Quote, Section B.2.3 OGBG-MolHIV); "Fine-tuning. The hyper-parameters for fine-tuning Graphormer on OGBG-MolHIV are presented in Table 9." (Quote, Section B.2.3 OGBG-MolHIV) |
| ZINC | No (trained separately) | No | Not specified. | "For benchmarking-GNN, which does not encourage large pre-trained model, we train an additional Graphormer<sub>SLIM</sub> (L=12, d=80, total param.= 489K) from scratch on ZINC." (Quote, Section 4.2 Graph Representation); "we train a slim 12-layer Graphormer with hidden dimension of 80, which is called Graphormer<sub>SLIM</sub> in Table 4" (Quote, Section B.2.4 ZINC) |

## 6. Input and Representation Constraints

- Fixed or variable input resolution: Not specified.
- Fixed patch size: Not specified.
- Fixed number of tokens: Not specified.
- Fixed dimensionality (e.g., strictly 2D): "2D molecular graphs" for PCQM4M-LSC (Quote, Section B.1 Details of Datasets: "given their 2D molecular graphs").
- Graph-structured input definition: "Let G=(V,E) denote a graph where  $V=\{v_1,v_2,\cdots,v_n\}$ , n=|V| is the number of nodes." (Quote, Section 2 Preliminary)
- Node feature assumption: "Let the feature vector of node  $v_i$  be  $x_i$ ." (Quote, Section 2 Preliminary)
- Non-grid structure: "there does not exist a canonical grid to embed the graph" (Quote, Section 3.1.2 Spatial Encoding).
- Spatial relation encoding depends on shortest-path distance: "In this paper, we choose  $\phi(v_i, v_j)$  to be the distance of the shortest path (SPD) between  $v_i$  and  $v_j$  if the two nodes are connected. If not, we set the output of  $\phi$  to be a special value, i.e., -1." (Quote, Section 3.1.2 Spatial Encoding)
- Centrality encoding adds degree embeddings at input: "To be specific, we develop a Centrality Encoding which assigns each node two real-valued embedding vectors according to its indegree and outdegree. As the centrality encoding is applied to each node, we simply add it to the node features as the input." (Quote, Section 3.1.1 Centrality Encoding)
- Special graph token: "we add a special node called [VNode] to the graph, and make connection between [VNode] and each node individually." (Quote, Section 3.2 Implementation Details of Graphormer)
- Edge features in attention bias: "For each ordered node pair  $(v_i, v_j)$ , we find (one of) the shortest path  $SP_{ij} = (e_1, e_2, ..., e_N)$  from  $v_i$  to  $v_j$ , and compute an average of the dot-products of the edge feature and a learnable embedding along the path." (Quote, Section 3.1.3 Edge Encoding in the Attention)

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Fixed or variable sequence length: Not specified.
- Attention type: Global. Evidence: "An advantage of Transformer is its global receptive field. In each Transformer layer, each token can attend to the information at any position" (Quote, Section 3.1.2 Spatial Encoding); "the Transformer layer provides a global information that each node can attend to all other nodes in the graph" (Quote, Section 3.1.2 Spatial Encoding).
- Mechanisms to manage computational cost: Not introduced; limitation noted. Evidence: "the quadratic complexity of the self-attention module restricts Graphormer's application on large graphs" (Quote, Section 6 Conclusion).

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: Bias-based relative spatial encoding based on shortest-path distance. Evidence: "In this paper, we choose  $\phi(v_i, v_j)$  to be the distance of the shortest path (SPD) between  $v_i$  and  $v_j$  if the two nodes are connected. If not, we set the output of  $\phi$  to be a special value, i.e., -1. We assign each (feasible) output value a learnable scalar which will serve as a bias term in the self-attention module." (Quote, Section 3.1.2 Spatial Encoding)
- Where it is applied: Attention bias in self-attention; shared across layers. Evidence: "We assign each (feasible) output value a learnable scalar which will serve as a bias term in the self-attention module." and " $b_{\phi(v_i,v_j)}$  is a learnable scalar indexed by  $\phi(v_i,v_j)$ , and shared across all layers." (Quote, Section 3.1.2 Spatial Encoding)
- Whether positional encoding is fixed across experiments or compared: Compared/ablated. Evidence: "We compare previously used positional encoding (PE) to our proposed spatial encoding" (Quote, Section 4.3 Ablation Studies).

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption: Treated as a research variable in ablations. Evidence: "We compare previously used positional encoding (PE) to our proposed spatial encoding" (Quote, Section 4.3 Ablation Studies).
- Multiple positional encodings compared: Yes; Laplacian PE is compared. Evidence: "We report the performance for Laplacian PE" (Quote, Section 4.3 Ablation Studies).
- Claim that PE choice is "not critical" or secondary: Not claimed.

## 10. Evidence of Constraint Masking

- Model sizes: "We primarily report results on two model sizes: Graphormer (L=12, d=768), and a smaller one Graphormer<sub>SMALL</sub> (L=6, d=512)." (Quote, Section 4.1 OGB Large-Scale Challenge)
- Dataset sizes: "PCQM4M-LSC is unprecedentedly large in scale comparing to other labeled graph-level prediction datasets, which contains more than 3.8M graphs." (Quote, Section B.1 Details of Datasets)
- Architectural encodings emphasized over scale: "Our key insight to utilizing Transformer in the graph is the necessity of effectively encoding the structural information of a graph into the model. To this end, we propose several simple yet effective structural encoding methods to help Graphormer better model graph-structured data." (Quote, Abstract); "Transformer architecture with the spatial encoding outperforms the counterpart built on the positional encoding, which demonstrates the effectiveness of using spatial encoding to capture the node spatial information." (Quote, Section 4.3 Ablation Studies); "Centrality Encoding. Transformer architecture with degree-based centrality encoding yields a large margin performance boost in comparison to those without centrality information." (Quote, Section 4.3 Ablation Studies).
- Scaling parameters alone not sufficient (baseline evidence): "we do not observe a performance gain along with the growth of parameters of GT." (Quote, Section 4.1 OGB Large-Scale Challenge)
- Training tricks: "we employ a widely used data augmentation for graph - FLAG [27], to mitigate the over-fitting problem on OGB datasets." (Quote, Section 4.2 Graph Representation)

## 11. Architectural Workarounds

- Centrality encoding: "To be specific, we develop a Centrality Encoding which assigns each node two real-valued embedding vectors according to its indegree and outdegree. As the centrality encoding is applied to each node, we simply add it to the node features as the input." (Quote, Section 3.1.1 Centrality Encoding)
- Spatial encoding as attention bias: "In this paper, we choose  $\phi(v_i, v_j)$  to be the distance of the shortest path (SPD) between  $v_i$  and  $v_j$  if the two nodes are connected. If not, we set the output of  $\phi$  to be a special value, i.e., -1. We assign each (feasible) output value a learnable scalar which will serve as a bias term in the self-attention module." (Quote, Section 3.1.2 Spatial Encoding)
- Edge encoding in attention: "To better encode edge features into attention layers, we propose a new edge encoding method in Graphormer. For each ordered node pair  $(v_i, v_j)$ , we find (one of) the shortest path  $SP_{ij} = (e_1, e_2, ..., e_N)$  from  $v_i$  to  $v_j$ , and compute an average of the dot-products of the edge feature and a learnable embedding along the path. The proposed edge encoding incorporates edge features via a bias term to the attention module." (Quote, Section 3.1.3 Edge Encoding in the Attention)
- Special node for graph readout: "Inspired by [15], in Graphormer, we add a special node called [VNode] to the graph, and make connection between [VNode] and each node individually. In the AGGREGATE-COMBINE step, the representation of [VNode] has been updated as normal nodes in graph, and the representation of the entire graph hG would be the node feature of [VNode] in the final layer." (Quote, Section 3.2 Implementation Details of Graphormer)

## 12. Explicit Limitations and Non-Claims

- Limitation on scalability: "the quadratic complexity of the self-attention module restricts Graphormer's application on large graphs." (Quote, Section 6 Conclusion)
- Future work needs: "future development of efficient Graphormer is necessary" and "an applicable graph sampling strategy is desired for node representation extraction with Graphormer." (Quote, Section 6 Conclusion)
- Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Molecular graphs only (PCQM4M-LSC, MolPCBA, MolHIV, ZINC).
> - Task structure: Multiple graph-level prediction tasks (classification and regression) across datasets.
> - Representation rigidity: Graph-structured inputs with shortest-path-distance bias and explicit node/edge feature encodings; no grid or patch constraints specified.
> - Model sharing vs specialization: Pretrain on PCQM4M-LSC and fine-tune for MolPCBA/MolHIV; separate training for ZINC.
> - Role of positional encoding: Spatial (shortest-path) attention bias is a key design and is compared against Laplacian PE.

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple graph-level prediction tasks across PCQM4M-LSC, OGBG-MolPCBA, OGBG-MolHIV, and ZINC, all described as molecular graph datasets (e.g., "we conduct experiments on two molecular graph datasets in popular OGB leaderboards, i.e., OGBG-MolPCBA and OGBG-MolHIV" and "We use the ZINC datasets, which is the most popular real-world molecular dataset"). It also uses a pretraining-and-fine-tuning setup within this molecular domain rather than cross-domain transfer.
