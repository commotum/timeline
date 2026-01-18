## 1. Basic Metadata

- Title: "Transformer based 3D tooth segmentation via point cloud region partition" (header)
- Authors: "You Wu"; "Hongping Yan"; "Kun Ding" (header)
- Year: "Published online: 18 November 2024" (Data availability)
- Venue (conference/journal/arXiv): "scientific reports" (header)

## 2. One-Sentence Contribution Summary

"In this paper, we propose a novel Transformer-based 3D tooth segmentation network, called PointRegion, which can process the entire point cloud at a low cost." (intro)

## 3. Tasks Evaluated

- Task name: 3D tooth segmentation (point-level semantic segmentation)
  - Task type: Segmentation
  - Dataset(s) used: "The entire dataset consists of 916 dental models (403 maxillaries and 513 mandibles), with each mesh model containing an average of 100,000 faces. We randomly and evenly split it into 815 models for training and 101 for evaluation." (Experimental evaluation Dataset and metrics)
  - Domain: "Automatic and accurate tooth segmentation on 3D dental point clouds plays a pivotal role in computer-aided dentistry." (intro); "Since our original dental model is mesh data, we need to convert it into point cloud." (Overview)
  - Task evidence: "learning-based 3D tooth segmentation is a crucial step for a computer-aided-design (CAD) system to automatically and accurately identify individual tooth and gingiva based on 3D real oral scanning data or dental model." (intro)

## 4. Domain and Modality Scope

- Single domain: Yes. "we collected a set of tooth mesh models from the real-world clinics" and "our dental dataset" (Experimental evaluation Dataset and metrics; intro).
- Multiple domains within the same modality: Not indicated; evaluation is only on dental data. "The entire dataset consists of 916 dental models" (Experimental evaluation Dataset and metrics).
- Multiple modalities: No; a single 3D point cloud modality derived from mesh. "Since our original dental model is mesh data, we need to convert it into point cloud." (Overview)
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| 3D tooth segmentation | Single task (N/A) | Not specified | Not specified | "Our segmentation networks are trained with PyTorch on a NVIDIA TITAN RTX 24GB GPU for 200 epochs" (Implementation details). |

## 6. Input and Representation Constraints

- Point cloud input with d-dimensional attributes: "Given an input mesh dental model, we sample it to obtain the point cloud consisting of N points, each of which has d-Dimensional attributes." (Mesh2Point)
- 3D coordinates as a base representation: "In the simplest case of d=3, each point is described by 3-Dimensional coordinates of the corresponding mesh cell center." (Mesh2Point)
- Optional added attributes and normalization: "it is also possible to include additional information such as the 3-Dimensional normal vector of the cell surface and the 9-Dimensional coordinates of the cell's three vertices" and "we perform min-max normalization along each dimension of the point cloud data." (Mesh2Point)
- Fixed-size sub-samples for training/testing: "We split all points into three sub-samples including 10240 points using FPS" and "we also use multiple FPS to get multiple sub-samples with the size of 10240" (Implementation details).
- Fixed region count in main experiments: "Our PointRegion employs the RegionPartition module to divide the point cloud into 1024 regions" and "Therefore, we set G to 1024 in the rest experiments." (intro; Ablative analysis)
- Fixed neighborhood size in ablations: "The number of nearest neighbor regions is fixed to 32." (Table 2 caption)

## 7. Context Window and Attention Structure

- Maximum sequence length: The attention sequence length equals the number of regions, with a main setting of 1024: "the RegionEncoder module accepts the 1024 region embeddings as the input sequence" (intro).
- Fixed or variable: Variable G in ablation but fixed to 1024 in main experiments: "we first fix K to 32 and vary G from 128 to 4096" and "Therefore, we set G to 1024 in the rest experiments." (Ablative analysis)
- Attention type: Global, offset-attention over all regions: "an offset-attention based RegionEncoder module is applied on all region embeddings to model global context among regions" (intro).
- Cost management mechanisms: "reducing the input sequence length" (intro) and "Since the number of regions is far less than the number of points, our proposed PointRegion model can leverage the capability of the global-based Transformer on large-scale point clouds with low computational cost and memory consumption." (intro)

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: Not specified.
- Where it is applied: Not specified.
- Fixed across experiments / modified / ablated: Not specified.

## 9. Positional Encoding as a Variable

- The paper does not treat positional encoding as a research variable, and no comparisons or ablations are stated: Not specified.

## 10. Evidence of Constraint Masking

- Dataset size: "The entire dataset consists of 916 dental models (403 maxillaries and 513 mandibles)" and "815 models for training and 101 for evaluation." (Experimental evaluation Dataset and metrics)
- Model size(s): "PointRegion w/o post-process | 2.490" (Table 1).
- Performance gains attributed to architecture rather than scale: "All the improvements are attributed to its partitioning strategy, which not only reduces input sequences, effectively reducing computational complexity, but also makes it easier for Transformer to learn differences between regions and get richer representations." (Experimental results)
- Scaling/sequence length effects: "Results in Table 2 show that the performance of our method can be improved as G increases" and "In the extreme case, where the number of regions is equal to the number of input point clouds (10240), we eliminate the branch of division of regions within the RegionPartition module and Point-to-Region Association mechanism, resulting in a significant increase in memory and computation compared to G=1024." (Ablative analysis)

## 11. Architectural Workarounds

- Non-overlapping region partitioning to reduce sequence length and ambiguity: "we propose an effective non-overlapping partitioning method in the RegionPartition module." (intro)
- Global offset-attention on region embeddings: "the RegionEncoder module... uses offset-attention mechanism... to learn the global context of the point cloud by directly modeling inter-region relations." (RegionEncoder module)
- Point-to-region association to map region logits to point-level labels for irregular point clouds: "a novel mechanism is designed to establish point-to-region association by utilizing information similarity between points and regions." (intro)
- Graph-cut post-processing to refine boundaries: "we use the graph-cut algorithm to post-process the segmentation results." (Point level segmentation based on point and region association)
- Sampling-based memory control: "Due to limitations of GPU memory, it is hard to input all points... We split all points into three sub-samples including 10240 points using FPS" (Implementation details)

## 12. Explicit Limitations and Non-Claims

- Limitations on boundary quality and rare cases: "segmentation of boundary details remains a challenge." and "Unsmooth segmentation boundaries and segmentation errors tends to be more common in cases involving extremely crowded teeth, dental calculus and swollen ginigiva." (Conclusions)
- Post-processing cost: "the graph-cut post-processing algorithm can effectively improve the segmentation details at the edges of teeth, the introduction of additional complexity means that more computational resources and time costs are required, which to some extent affects the practicality of the method." (Conclusions)
- Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)

**Constraint Profile:**
- Domain scope: Single dental domain (clinical dental meshes/point clouds).
- Task structure: Single semantic segmentation task (tooth/gingiva segmentation).
- Representation rigidity: Mesh converted to point cloud; training uses fixed 10240-point sub-samples; region sequence length fixed to G=1024 with K=32 in main experiments.
- Model sharing vs specialization: Single model trained for one task; no multi-task sharing discussed.
- Role of positional encoding: Not described.

### 14. Final Classification

**Single-task, single-domain.** The paper evaluates only 3D tooth segmentation on a single dental dataset of "916 dental models" with a single training/evaluation split (Experimental evaluation Dataset and metrics). All experiments focus on dental point clouds/meshes with no additional tasks or domains described.
