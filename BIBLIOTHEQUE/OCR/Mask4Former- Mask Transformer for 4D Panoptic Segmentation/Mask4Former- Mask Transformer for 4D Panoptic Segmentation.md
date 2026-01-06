## Mask4Former: Mask Transformer for 4D Panoptic Segmentation

Kadir Yilmaz<sup>1</sup>, Jonas Schult<sup>1</sup>, Alexey Nekrasov<sup>1</sup>, Bastian Leibe<sup>1</sup>

Abstract—Accurately perceiving and tracking instances over time is essential for the decision-making processes of autonomous agents interacting safely in dynamic environments. With this intention, we propose Mask4Former for the challenging task of 4D panoptic segmentation of LiDAR point clouds. Mask4Former is the first transformer-based approach unifying semantic instance segmentation and tracking of sparse and irregular sequences of 3D point clouds into a single joint model. Our model directly predicts semantic instances and their temporal associations without relying on hand-crafted non-learned association strategies such as probabilistic clustering or voting-based center prediction. Instead, Mask4Former introduces spatio-temporal instance queries that encode the semantic and geometric properties of each semantic tracklet in the sequence. In an in-depth study, we find that promoting spatially compact instance predictions is critical as spatiotemporal instance queries tend to merge multiple semantically similar instances, even if they are spatially distant. To this end, we regress 6-DOF bounding box parameters from spatiotemporal instance queries, which are used as an auxiliary task to foster spatially compact predictions. Mask4Former achieves a new state-of-the-art on the SemanticKITTI test set with a score of 68.4 LSTQ.

#### I. INTRODUCTION

LiDAR is a popular sensor modality in the robotics community due to its ability to provide accurate 3D spatial information. It allows precise scene understanding of the 3D environment over time, which is essential for agents to safely navigate in dynamic environments by predicting traffic movements and identifying potential hazards. To achieve the full potential of LiDAR data, in this work, we address the task of 4D panoptic segmentation. That is, given a sequence of LiDAR scans, the goal is to predict the semantic class of each point while consistently tracking object instances. The research community has made remarkable progress in advancing 3D vision tasks, fueled by the rapid advancement of deep learning methods [29, 38, 51] and the availability of large-scale benchmark datasets [5, 15, 16, 47]. Powerful feature extractors [11, 49, 51, 62] that exploit the rich information offered by LiDAR sensors have been proposed, leading to remarkable improvements in object detection [24, 42, 56], segmentation [32, 51, 62], and tracking [55, 58].

To accomplish holistic 3D scene understanding, 4D panoptic segmentation [2] has recently attracted attention. Traditionally, approaches follow the tracking-by-detection paradigm [35] which decouples 4D panoptic segmentation in the subtasks of semantic segmentation [32, 51], object detection [24] and tracking [34, 55]. While this separation of segmentation, detection, and tracking allows for independent

<sup>1</sup>Computer Vision Group, RWTH Aachen University, Germany. Project Page: https://vision.rwth-aachen.de/Mask4Former

![](_page_0_Figure_8.jpeg)

Fig. 1: **Spatially non-compact instances.** Naively adapted for 4D panoptic segmentation, mask transformer approaches reveal a crucial shortcoming: instance predictions tend to be spatially non-compact. As a result, the baseline model predicts two cars as a single object (*left*). To overcome this limitation, we introduce Mask4Former, which additionally regresses 6-DOF bounding box parameters for the instance trajectory. We find that optimizing these bounding box parameters provides a valuable loss signal that promotes spatially compact instances (*right*).

improvements in each component, it tends to neglect joint learning of temporal relationships with semantic information. Significant advances in 4D panoptic segmentation methods address this problem by introducing model architectures that approach the task as a whole and predict semantic class labels for each point and temporally consistent instances [1]. Recent methods generate instance predictions by grouping proposals in the 4D spatio-temporal volume [2, 18, 22] or learned embedding space [31]. However, all previous 4D panoptic segmentation methods fundamentally rely on non-learned clustering methods to aggregate tracklets.

At the same time, we observe a noticeable shift towards unifying tasks [21,52,57] and model architectures [7,10] for holistic scene understanding. Central to this trend are mask transformers [8,9,38] that directly predict foreground masks and their associated semantic labels, eliminating the need for non-learned clustering strategies. Typically, these models consist of two main components: a convolutional feature extractor and a transformer decoder. The convolutional feature extractor processes the point cloud and generates multi-scale features. The transformer decoder leverages these extracted features and iteratively refines queries each of

which encodes the spatial and semantic features for an instance. Throughout multiple transformer decoder layers, the queries are refined sequentially. Ultimately, these refined queries directly predict the final semantic class and mask predictions, allowing mask transformers to avoid hand-crafted grouping. Despite the remarkable performance of mask transformer architectures across diverse tasks, such as image segmentation [9, 10], video segmentation [8], and 3D scene segmentation [29, 38, 46], it remains open whether such a paradigm generalizes to the unique challenges of 4D panoptic segmentation of sparse point cloud sequences.

To answer this question, our goal in this paper is to extend mask transformers to 4D panoptic segmentation of point clouds. Unlike prevailing top-performing approaches for 4D panoptic segmentation [2, 22, 31, 61], we directly predict foreground masks for thing instances and stuff regions and their associated semantic labels, bypassing the need for post-processing clustering which requires hand-engineered methods and fine-tuned hyperparameters. Therefore, in an initial study, we adapt Mask3D [38] for 4D panoptic segmentation. We follow recent approaches [2, 19, 22] by superimposing consecutive LiDAR scans into a spatio-temporal point cloud that is processed by a sparse convolutional feature backbone [11]. Furthermore, we introduce pointwise spatio-temporal positional encoding in the transformer decoder [8]. Our findings indicate that these modifications are already competitive with specialized 4D panoptic segmentation methods [22]. Yet, a deeper examination reveals a significant flaw in mask transformer approaches for 3D point clouds: instances are not always spatially compact [38, 40]. Specifically, an instance query may connect multiple instances in the spatio-temporal point cloud, even if they are spatially distant but share semantic similarities (Fig. 1, *left*).

Based on these findings, we introduce our novel approach called Mask4Former, which is tailored to ensure spatially compact instances, thus unleashing the full potential of mask transformer architectures for 4D panoptic segmentation. We achieve this by regressing 6-DOF bounding box parameters from the spatio-temporal queries, providing a loss signal to foster spatially compact instance predictions (Fig. 1, *right*). We evaluate our Mask4Former model on the challenging SemanticKITTI 4D panoptic segmentation benchmark and achieve state-of-the-art performance on the test set.

In summary, our contributions are fourfold: (1) We extend the state-of-the-art instance segmentation method Mask3D [38] to the 4D panoptic segmentation task. (2) In experiments, we discover a crucial shortcoming of this straightforward adaptation, namely, the tendency for spatio-temporal instance predictions to lack spatial compactness. (3) We propose Mask4Former which effectively addresses the aforementioned limitation by introducing a box regression branch that promotes spatially compact instance predictions in an end-to-end trainable fashion, rather than relying on a geometric grouping mechanism with hand-tuned hyperparameters. (4) Mask4Former achieves state-of-the-art performance on the SemanticKITTI 4D panoptic segmentation benchmark.

#### II. RELATED WORK

**Mask Transformers.** MaskFormer [10] proposes mask classification as a novel segmentation technique, showcasing its advantages over conventional pixel-based methods. Inspired by DETR [7], it combines CNNs and transformer networks in a universal segmentation architecture, eliminating the need for task-specific architectures, and streamlining development processes. Subsequently, Mask2Former [9] introduces masked attention in the transformer decoder, directing the attention only to relevant parts of the image, and incorporates high-resolution multi-scale features for segmenting smaller objects. This improves convergence and performance, achieving state-of-the-art results in 2D segmentation tasks [21, 27, 59]. The paradigm extends to the video instance segmentation [8] task, where Mask2Former effectively addresses temporal consistency, showcasing its universal applicability. Inspired by its success in 2D, Mask3D [38] applies the mask transformer architecture to the 3D domain by leveraging a sparse convolutional backbone [11], and eliminates the need for the predominantly used centervoting and clustering algorithms [14, 20, 53]. For LiDAR panoptic segmentation, MaskPLS [29] compares mask transformer architectures with adapted semantic segmentation approaches [6, 11, 12, 18, 49, 62], demonstrating the superiority of the mask transformer architecture.

**4D** panoptic segmentation. 4D-PLS [2] introduces the 4D panoptic segmentation task, associated evaluation metrics, and their method for solving the task. It superimposes consecutive LiDAR scans to form a spatio-temporal point cloud, performs semantic segmentation, and follows a probabilistic approach for clustering instances based on their predicted centers. Along the same lines, 4D-DS-Net [19] and 4D-StOP [22] propose to cluster instances based on spatiotemporal proximity. 4D-DS-Net [19] extends DS-Net [18] to the 4D domain by applying a dynamic shifting module to spatio-temporal point clouds which iteratively refines the estimated instance centers and clusters the points in the spatiotemporal volume. 4D-StOP [22], on the other hand, replaces the probabilistic clustering with an instance-centric voting approach. Here, initial instance proposals are generated using center votes and then aggregated using learned geometric features. Building on the success of 4D-StOP, the concurrent work Eq-4D-StOP [61] predicts equivariant fields and incorporates the necessary layers into the models. This reinforcement of rotation equivariance ensures that the models account for rotational symmetries in the data, resulting in a more robust feature learning. Contrastingly, CA-Net [31] clusters instances in the feature space. It leverages an off-the-shelf 3D panoptic segmentation network [18] and uses extracted point features in a contrastive learning framework [17] to generate instance-wise consistent features, resulting in robust instance associations over time. Bypassing the need for non-learned clustering approaches, the concurrent work Mask4D [30] adopts the mask transformer-based paradigm but opts for queries that encode single frame instances, and re-uses these queries in subsequent frames to facilitate tracking. Unlike

![](_page_2_Figure_0.jpeg)

Fig. 2: **Illustration of the Mask4Former model.** We superimpose a sequence of T point clouds into a spatio-temporal representation which is subsequently processed by a sparse convolutional feature backbone  $\square$ . Given a multi-scale feature representation extracted from the feature backbone, the transformer decoder  $\square$  iteratively refines spatio-temporal (ST) instance queries. A mask module  $\square$  consumes ST queries and point features at various scales and predicts semantic class probabilities, instance heatmaps, and a 6-DOF bounding box for each ST query.

previous approaches, Mask4Former unifies segmentation and tracking by directly predicting the spatio-temporal instance masks and their corresponding semantic labels.

### III. METHOD

Inspired by the success of mask transformer approaches for 3D instance segmentation [29, 38, 46] and 2D video instance segmentation [8], we propose Mask4Former – the first mask transformer-based approach for 4D panoptic segmentation. Building on Mask3D [38] for 3D instance segmentation, we introduce technical components that are key to enabling 4D panoptic segmentation of point clouds, *i.e.*, predicting the semantic class of each point and consistently tracking instances over time.

Overview. (Fig. 2) As the input to our model, we use a single voxelized point cloud consisting of superimposed consecutive LiDAR scans. We process the point cloud with a sparse convolutional feature extractor (Fig.2,  $\square$ ), which generates a multi-resolution voxel representation for the transformer  $decoder \square$ . At the core of the model are spatio-temporal (ST) queries that encode geometric and semantic attributes of all instances in a sequence. To learn ST query features, we use a transformer decoder that encompasses consecutive query refinement and mask modules. A *mask module* takes the ST queries and predicts instance heatmaps, semantic class probabilities, and also regresses a bounding box for each instance trajectory. A query refinement module □ updates the ST queries by cross-attending to multi-scale voxel representations. In the following, we provide a detailed description of each component involved.

Input Spatio-Temporal Point Cloud. We represent a temporal sequence of point clouds as a single superimposed and voxelized point cloud. Similar to other approaches [2, 22], we use pose estimates of the ego vehicle [3, 4] to create a single scene containing points from multiple LiDAR scans in a global coordinate frame. Subsequently, this superimposed point cloud represents a spatio-temporal volume, denoted as  $\mathcal{P} \in \mathbb{R}^{M \times 3}$ , which captures the temporal evolution of the scene. We partition this point cloud into equally sized cubic voxels, thus yielding the representation  $\mathcal{V} \in \mathbb{Z}^{K_0 \times 3}$ . This voxelization process not only keeps memory constraints in bounds but also allows for efficient processing of the resulting point cloud by sparse convolutional extractors [11].

**Feature Backbone.** (Fig. 2,  $\square$ ) The sparse convolutional feature extractor processes the voxelized point cloud  $\mathcal{V} \in \mathbb{Z}^{K_0 \times 3}$  and extracts multi-scale features  $F_r \in \mathbb{R}^{K_r \times D_r}$  at various resolutions r. This design allows the network to capture both local geometry and global context while ensuring the preservation of fine-grained spatial details.

**Mask Module.** (Fig. 2,  $\square$ ) Each of the  $N_q$  ST queries  $\mathbf{X} \in \mathbb{R}^{N_q \times D}$  represents a distinct instance over a time period. The mask module predicts the foreground mask of an instance throughout the sequence and the semantic class of the mask, as well as estimating the 6-DOF bounding box parameters of its trajectory. To generate this binary foreground mask, ST queries are processed by an MLP, and aligned with the feature space of the backbone's output. To obtain spatio-temporal masks at the finest resolution, we compute the dot product with the finest backbone features  $\mathbf{F}_0$ , which – after sigmoid activation and thresholding – yields the final binary ST mask. In addition to these masks, we predict

semantic class probabilities for each ST query via a linear projection layer to C+1 dimensions, followed by a softmax normalization. A critical element for consistent tracking of instances over time is the bounding box regression branch. We feed the ST queries to an MLP followed by sigmoid activation to map the features to a 6-dimensional bounding box parameter space that encodes the normalized bounding box center coordinates (x,y,z) as well as the box dimensions (w,h,d).

**Query Refinement Module.** (Fig. 2,  $\square$ ) Following Cheng *et al.* [9], the query refinement blocks refine the ST queries **X** by using the voxel features  $\mathbf{F}_r$  at various resolutions r. First, a masked cross-attention layer [9] transforms voxel features  $\mathbf{F}_r$  into keys K and values V, while ST queries are mapped to queries Q. Here, ST queries attend only to the foreground voxels predicted by the previous mask module. We then apply self-attention between queries to ensure that multiple queries do not converge on a single instance.

We use spatio-temporal Fourier positional encodings [48] to incorporate both spatial and temporal information into our transformer blocks. To do this, we sum spatial positional encodings based on the voxel positions and temporal positional encodings based on the LiDAR scan time frame [8].

**Hungarian Matching.** (Fig. 2,  $\square$ ) In a single forward pass, Mask4Former determines  $N_q$  foreground masks along with their associated semantic class labels. Since both these predictions and the ground truth targets are not in any particular order, it is necessary to establish optimal one-to-one correspondences between them for model optimization. Typically, Mask transformer methods [7,9,10] rely on the Hungarian Algorithm [23] for this purpose. The assignment cost for a predicted semantic mask, *i.e.*, thing instances and stuff regions, and a target mask is defined as follows:

$$C = \mathcal{L}_{\text{mask}} + \mathcal{L}_{\text{sem}} \tag{1}$$

where  $\mathcal{L}_{mask} = \lambda_{dice} \mathcal{L}_{dice} + \lambda_{BCE} \mathcal{L}_{BCE}$  is a weighted combination of the binary cross-entropy loss and the dice loss [33] for supervising foreground mask predictions and  $\mathcal{L}_{sem} = \lambda_{CE} \mathcal{L}_{CE}$  is the multi-class cross-entropy loss  $\mathcal{L}_{CE}$  for supervising mask semantics. The Hungarian algorithm is applied to solve the assignment problem and to find the globally optimal matching that minimizes the total cost while ensuring that each target mask is assigned only once. The unmatched predicted masks are assigned to a "no-object" mask.

Training the model. After establishing one-to-one correspondences, we can directly optimize each predicted mask. Our resulting loss consists of three loss functions: We keep the same binary mask loss  $\mathcal{L}_{mask}$  and the multi-class crossentropy loss  $\mathcal{L}_{sem}$  from the Hungarian matching as referenced in Eq. 1. Observing that the  $\mathcal{L}_{mask}$  loss does not consider the distance of incorrectly added points to the mask, we introduce a new auxiliary bounding box regression loss  $\mathcal{L}_{box}$  which promotes spatially compact instances. We implement the bounding box loss as an L1 loss on the normalized axisaligned box parameters. By optimizing the bounding box parameters from ST queries, the spatial location of their

corresponding masks is supervised. Consequently, this helps to distinguish similar instances of the same class that are spatially separated. The overall loss is:

$$\mathcal{L} = \mathcal{L}_{\text{mask}} + \mathcal{L}_{\text{sem}} + \mathcal{L}_{\text{box}} \tag{2}$$

Extracting 4D panoptic segmentations. Mask4Former predicts  $N_q$  instance tracks as semantic heatmaps which are not necessarily non-overlapping. To assign a single semantic class label and instance ID to every point within the spatio-temporal point cloud, we proceed in the following manner: First, for each spatio-temporal query, we obtain semantic confidence by selecting the semantic class with the maximum probability. Second, this semantic confidence is multiplied with the corresponding instance heatmap, resulting in an overall confidence heatmap. We then assign each point to the query with the maximum confidence.

**Tracking over long sequences.** To track instances across long LiDAR sequences that exceed memory limits, it is critical to associate instances across successive spatio-temporal point clouds. Therefore, we follow Aygün *et al.* [2] and construct long sequences from short sequences in a way that ensures seamless associations. We establish a one-to-one match between predicted instances in the last and first frames between short sequences.

#### IV. EXPERIMENTS

A. Comparing with State-of-the-Art Methods.

**Dataset.** We evaluate Mask4Former on the well-established SemanticKITTI dataset [3], which is derived from the KITTI odometry dataset [16]. The dataset is split into training, validation, and test sets, and consists of over 43,000 LiDAR scans recorded with a Velodyne-64 laser scanner capturing various urban driving scenarios. Each point in the LiDAR point clouds is densely annotated with one of C=19 semantic labels, e.g., car, road, cyclist, as well as a unique instance ID that is consistent over time. For every time step, the dataset includes precise pose estimates of the ego vehicle, which is critical for the 4D panoptic segmentation task.

Metric. The LiDAR Segmentation and Tracking Quality Metric (LSTO) [2] is designed to evaluate the performance of 4D panoptic segmentation algorithms. It consists of two main components: classification and association scores. The classification score  $S_{cls}$  evaluates how well the algorithm performs in assigning correct semantic labels to the LiDAR points. It is calculated as the instance-agnostic mean intersection over union (mIoU) over all classes. The association score  $S_{assoc}$  evaluates the quality of point-to-instance associations considering the entire LiDAR sequence. It measures how well the algorithm tracks object instances over time without considering the semantic predictions. The overall LSTQ metric is computed as the geometric mean of the classification score and the association score: LSTQ = $\sqrt{S_{cls} \times S_{assoc}}$ . The geometric mean ensures that a high score can only be obtained if the approach performs well in both the classification and the association task.

TABLE I: **Scores on SemanticKITTI test.** SemanticKITTI 4D panoptic segmentation test set results. *Abbreviations:* PP: PointPillars [24], MOT: Multi-Object Tracking [55], SFP: Scene flow based propagation [34]. \* denotes concurrent work.

| Method                 | LSTQ        | Sassoc | $S_{cls}$   | ${\rm Io}{\bf U}^{St}$ | $IoU^{Th}$ |
|------------------------|-------------|--------|-------------|------------------------|------------|
| KPConv [51]+PP+MOT     | 38.0        | 25.9   | 55.9        | 66.9                   | 47.7       |
| RangeNet++ [32]+PP+SFP | 34.9        | 23.3   | 52.4        | 64.5                   | 35.8       |
| KPConv [51]+PP+SFP     | 38.5        | 26.6   | 55.9        | 66.9                   | 47.7       |
| 4D-PLS [2]             | 56.9        | 56.4   | 57.4        | 66.9                   | 51.6       |
| 4D-DS-Net [19]         | 62.3        | 65.8   | 58.9        | 65.6                   | 49.8       |
| CIA [31]               | 63.1        | 65.7   | 60.6        | 66.9                   | 52.0       |
| 4D-StOP [22]           | 63.9        | 69.5   | 58.8        | 67.7                   | 53.8       |
| Mask4D* [30]           | 64.3        | 66.4   | 62.2        | 69.9                   | 52.2       |
| Eq-4D-StOP* [61]       | <u>67.8</u> | 72.3   | <u>63.5</u> | <u>70.4</u>            | 61.9       |
| Mask4Former (Ours)     | 68.4        | 67.3   | 69.6        | 72.7                   | 65.3       |

TABLE II: Scores on SemanticKITTI validation. \* denotes concurrent work.

| Method                 | LSTQ        | Sassoc      | $S_{cls}$   | $\mathrm{IoU}^{St}$ | $IoU^{Th}$ |
|------------------------|-------------|-------------|-------------|---------------------|------------|
| KPConv [51]+PP+MOT     | 46.3        | 37.6        | 57.0        | 64.2                | 54.1       |
| RangeNet++ [32]+PP+SFP | 43.4        | 35.7        | 52.8        | 60.5                | 42.2       |
| KPConv [51]+PP+SFP     | 46.0        | 37.1        | 57.0        | 64.2                | 54.1       |
| 4D-PLS [2]             | 62.7        | 65.1        | 60.5        | 65.4                | 61.3       |
| 4D-StOP [22]           | 67.0        | 74.4        | 60.3        | 65.3                | 60.9       |
| 4D-DS-Net [19]         | 68.0        | 71.3        | 64.8        | 64.5                | 65.3       |
| Eq-4D-StOP* [61]       | 70.1        | 77.6        | 63.4        | 66.4                | 67.1       |
| Mask4D* [30]           | 71.4        | <u>75.4</u> | 67.5        | 65.8                | 69.9       |
| Mask4Former (Ours)     | <u>70.5</u> | 74.3        | <u>66.9</u> | 67.1                | 66.6       |

Implementation Details. In all experiments, we use  $N_q{=}100~{\rm ST}$  queries which are initialized with Farthest Point Sampled (FPS) point positions [36, 38]. Each spatiotemporal point cloud is formed by superimposing 2 consecutive LiDAR scans which are voxelized with a voxel size of 5 cm. The sparse feature backbone is a Minkowski Res16UNet34C [11]. We train the model for 30 epochs with a batch size of 4 using the AdamW optimizer [28] and the one-cycle learning rate scheduler [44] with a maximum learning rate of  $2 \cdot 10^{-4}$ . We perform standard data augmentation techniques including random rotation, translation, scaling, and instance population [56]. For the test set submission, we employ random rotation and translation as test time augmentations to enhance the semantic predictions.

**Results.** In Tables I and II, we report the scores on the SemanticKITTI 4D panoptic segmentation test and validation set, respectively. Mask4Former outperforms previous approaches by at least +4.5 LSTQ on the test set and +2.5 LSTQ on the validation set. Notably, Mask4Former demonstrates strong semantic understanding by achieving at least +9.0  $S_{cls}$  improvement over previous methods on the test set.

#### B. Analysis Experiments.

**Spatio-Temporal Formation.** We achieve a globally consistent sequence of LiDAR scans by leveraging the precise pose estimates from the LiDAR sensor [4]. Considering that the sparse convolutional feature backbone (Fig. 2, □) can process 3- and 4-dimensional inputs [11], we investigate

TABLE III: **Spatio-Temporal Formation.** We compare 3 different strategies for representing LiDAR point cloud sequences. We observe that it is key to enable the feature backbone to incorporate temporal information in the feature representation by creating a 4D spatio-temporal representation or superimposing 3D scans, leading to association improvements of up to  $+7.8\,\mathrm{S}_{\mathrm{assoc}}$ .

| Feature Extraction                      | LSTQ         | Sassoc       | $S_{cls}$    | $\mathrm{IoU}^{St}$ | $IoU^{Th}$   |
|-----------------------------------------|--------------|--------------|--------------|---------------------|--------------|
| ① Sequential 3D<br>② Spatio-temporal 4D | 64.3<br>68.8 | 65.8<br>72.6 | 62.8<br>65.2 | 64.0<br>66.0        | 61.2<br>64.1 |
| ③ Superimposed 3D                       | 70.2         | 73.6         | 66.9         | 67.2                | 66.5         |

TABLE IV: Ablation study on bounding box regression. We observe that optimizing Mask4Former using the regressed bounding box parameters leads to substantially better association scores compared to the baseline  $(+3.5\,\mathrm{S}_{\mathrm{assoc}})$ .

|   | $\mathcal{L}_{box}$ | DBS | LSTQ | $S_{cls}$ | Sassoc                                   |
|---|---------------------|-----|------|-----------|------------------------------------------|
| 1 | Х                   | ×   | 68.6 | 67.3      | 70.1                                     |
| 2 | X                   | ✓   | 70.1 | 67.3      | 70.1———————————————————————————————————— |
| 3 | ✓                   | X   | 70.2 | 66.9      | 73.6≪                                    |
| 4 | ✓                   | ✓   | 70.5 | 66.9      | <b>74.3</b> ←+4.                         |

which representation is best for extracting meaningful spatiotemporal features from a sequence. In Table III, we explore 3 different strategies for representing spatio-temporal feature volumes. Similar to Cheng et al. [8], in the first option (1), we process each LiDAR frame individually and then concatenate them along the spatial dimension before passing them to the Transformer decoder. In the second option (2), we represent a LiDAR sequence as a 4D feature volume, which is fed into a 4D sparse convolutional feature backbone [11], facilitating the learning of both spatial and temporal relationships directly within the backbone. Incorporating temporal data early in the backbone shows significant improvements in association quality, yielding an increase of +6.8 S<sub>assoc</sub>. Given the inherent sparsity of point clouds, the third approach (3) superimposes, i.e. concatenates, several point clouds into a single 3D volume [2,22]. We suspect that superimposing LiDAR scans leads to a denser representation, that is less susceptible to noise, yielding the best performance (Tab. III).

Spatially non-compact instance predictions. Achieving consistent tracking of multiple instances over time in LiDAR sequences is particularly challenging. This is due to the sparsity of the point clouds, as well as the occlusions and deformations that instances undergo over time, requiring robust temporal feature learning. In an initial study, we analyze our baseline method without the bounding box regression branch in the mask module (Fig. 2, □ and Tab. IV, ①), which reveals a crucial shortcoming of applying mask-transformer approaches directly to the task of 4D panoptic segmentation: Instance predictions tend to lack spatial compactness, *i.e.*, the spatio-temporal queries group multiple instances with similar semantics together, even if they are spatially distant (Fig. 1, *left*). To validate this observation, we apply the density-

![](_page_5_Figure_0.jpeg)

Fig. 3: **Visualization of learned point representations.** We use PCA to project the learned point representation of instances into RGB space. Our model trained without bounding box supervision, exhibits reduced variance in its feature representation for instances. In contrast, Mask4Former effectively separates distinct instances in the feature space.

based clustering method, DBSCAN [6], to each foreground mask prediction. This separates the instance mask predictions into spatially compact instances. The impact was noticeable: applying DBSCAN (2) to the instance predictions results in a significant improvement of +2.7 S<sub>assoc</sub>, confirming our initial findings and supporting our hypothesis. Anticipating further improvements by replacing DBSCAN with a learned component, we introduce a specialized box regression branch (3) which promotes spatial awareness to better separate instances. This approach outperforms the baseline, both with and without DBSCAN, by a margin of up to  $+3.5 \, S_{assoc}$ . Combining the box regression branch with DBSCAN yields our proposed method Mask4Former (4), which not only ensures a strong association between instances  $(+4.2 \, S_{assoc})$  but also achieves strong semantic scene understanding, scoring 66.9 S<sub>cls</sub> on the SemanticKITTI validation.

Visualization of point features learned by Mask4Former. In Fig. 3, we show examples of PCA projected features  $F_0$  extracted from the finest resolution of Mask4Former's feature backbone (Fig. 2,  $\square$ ). When trained without our suggested box loss, Mask4Former shows less distinct separation of instance point features within the feature space (Fig. 3a). Conversely, the model optimized with the auxiliary task of 6-DOF bounding box regression for each instance trajectory shows a distinct separation of instance point features in the feature space (Fig. 3b). This indicates that Mask4Former learns a more semantically meaningful feature space for the task of 4D panoptic segmentation leading to its superior association score  $S_{\rm assoc}$ , as highlighted in Tab. IV.

Qualitative results. In Fig. 4a, we show qualitative results. We observe that Mask4Former not only produces sharp instance masks but also reliably tracks the moving bicyclist throughout the entire sequence. We also demonstrate a failure case of our tracking approach. As we process long sequences by stitching short sequences with overlaps, we incorrectly split tracks when an instance is not present in the overlapping

![](_page_5_Figure_5.jpeg)

Fig. 4: **Qualitative Results.** We show color-coded instance tracks over 8 superimposed frames in a spatio-temporal point cloud and a failure where a pedestrian track is split due to an observation being outside of the LiDAR's field of view.

LiDAR scan. For example, in Fig. 4b, a pedestrian near the ego vehicle falls below the LiDAR's field of view. As a result, when the pedestrian becomes visible again, our tracking approach fails and predicts it as a new instance.

### V. CONCLUSION

Inspired by the success of recent mask transformer-based approaches, we have extended Mask3D to the task of 4D panoptic segmentation and have achieved promising results. In an in-depth analysis, we have found that Mask3D for 4D panoptic segmentation tends to produce spatially noncompact instances, resulting in poor association quality. To overcome this limitation, we have introduced Mask4Former, the first transformer-based approach, that unifies segmentation and tracking of 3D point cloud sequences and is tailored to ensure spatially compact instances. To this end, Mask4Former regresses 6-DOF bounding box parameters that are optimized to provide a loss signal to encourage spatially compact instance predictions. Through extensive experimental evaluations, we have demonstrated the effectiveness of Mask4Former, achieving state-of-the-art performance on the SemanticKITTI 4D panoptic segmentation benchmark. We anticipate follow-up work along the lines of direct prediction of instance and semantic labels.

**Acknowledgments:** This project is partially funded by the Bosch-RWTH LHC project "Context Understanding for Autonomous Systems", the BMBF project 6GEM (16KISK036K) and the NRW project WestAI (01IS22094D). Compute resources were granted by RWTH Aachen under project supp0003. This work is part of the first author's master thesis.

#### REFERENCES

- Ali Athar, Enxu Li, Sergio Casas, and Raquel Urtasun. 4D-Former: Multimodal 4D Panoptic Segmentation. In Conference on Robot Learning, 2023.
- [2] Mehmet Aygun, Aljosa Osep, Mark Weber, Maxim Maximov, Cyrill Stachniss, Jens Behley, and Laura Leal-Taixé. 4D Panoptic Lidar Segmentation. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2021.
- [3] Jens Behley, Martin Garbade, Andres Milioto, Jan Quenzel, Sven Behnke, Cyrill Stachniss, and Jurgen Gall. SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences. In *Interna*tional Conference on Computer Vision, 2019.
- [4] Jens Behley and Cyrill Stachniss. Efficient Surfel-Based SLAM using 3D Laser Range Data in Urban Environments. In Robotics: Science and Systems, 2018.
- [5] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. NuScenes: A Multimodal Dataset for Autonomous Driving. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2020.
- [6] Ricardo JGB Campello, Davoud Moulavi, and Jörg Sander. Density-based Clustering based on Hierarchical Density Estimates. In Pacific-Asia Conference on Knowledge Discovery and Data Mining, 2013.
- [7] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-End Object Detection with Transformers. In European Conference on Computer Vision, 2020.
- [8] Bowen Cheng, Anwesa Choudhuri, Ishan Misra, Alexander Kirillov, Rohit Girdhar, and Alexander G Schwing. Mask2Former for Video Instance Segmentation. arXiv:2112.10764, 2021.
- [9] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention Mask Transformer for Universal Image Segmentation. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2022.
- [10] Bowen Cheng, Alex Schwing, and Alexander Kirillov. Per-Pixel Classification is Not All You Need for Semantic Segmentation. In Neural Information Processing Systems, 2021.
- [11] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4D Spatio-Temporal Convnets: Minkowski Convolutional Neural Networks. In IEEE Conference on Computer Vision and Pattern Recognition, 2019.
- [12] Dorin Comaniciu and Peter Meer. Mean Shift: A Robust Approach Toward Feature Space Analysis. In *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2002.
- [13] Fabian Duerr, Mario Pfaller, Hendrik Weigel, and Jürgen Beyerer. Lidar-based Recurrent 3D Semantic Segmentation with Temporal Memory Alignment. In *International Conference on 3D Vision*, 2020.
- [14] Francis Engelmann, Martin Bokeloh, Alireza Fathi, Bastian Leibe, and Matthias Nießner. 3D-MPA: Multi Proposal Aggregation for 3D Semantic Instance Segmentation. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2020.
- [15] Whye Kit Fong, Rohit Mohan, Juana Valeria Hurtado, Lubing Zhou, Holger Caesar, Oscar Beijbom, and Abhinav Valada. Panoptic nuScenes: A Large-Scale Benchmark for LiDAR Panoptic Segmentation and Tracking. In *IEEE Robotics and Automation Letters (RA-L)*, 2022.
- [16] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite. In IEEE Conference on Computer Vision and Pattern Recognition, 2012.
- [17] Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality Reduction by Learning an Invariant Mapping. In *IEEE Conference* on Computer Vision and Pattern Recognition, 2006.
- [18] Fangzhou Hong, Hui Zhou, Xinge Zhu, Hongsheng Li, and Ziwei Liu. LiDAR-based Panoptic Segmentation via Dynamic Shifting Network. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2021.
- [19] Fangzhou Hong, Hui Zhou, Xinge Zhu, Hongsheng Li, and Ziwei Liu. LiDAR-based 4D Panoptic Segmentation via Dynamic Shifting Network. arXiv:2203.07186, 2022.
- [20] Li Jiang, Hengshuang Zhao, Shaoshuai Shi, Shu Liu, Chi-Wing Fu, and Jiaya Jia. PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation. In *IEEE Conference on Computer Vision and Pattern Recognition*. 2020.
- [21] Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr Dollár. Panoptic Segmentation. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2019.

- [22] Lars Kreuzberg, Idil Esen Zulfikar, Sabarinath Mahadevan, Francis Engelmann, and Bastian Leibe. 4D-StOP: Panoptic Segmentation of 4D LiDAR using Spatio-temporal Object Proposal Generation and Aggregation. In European Conference on Computer Vision Workshop, 2022.
- [23] Harold W. Kuhn. The Hungarian Method for the Assignment Problem. In Naval Research Logistics, 1955.
- [24] Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. PointPillars: Fast Encoders for Object Detection from Point Clouds. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2018.
- [25] Enxu Li, Sergio Casas, and Raquel Urtasun. MemorySeg: Online Lidar Semantic Segmentation with a Latent Memory. In *International Conference on Computer Vision*, 2023.
- [26] Jinke Li, Xiao He, Yang Wen, Yuan Gao, Xiaoqiang Cheng, and Dan Zhang. Panoptic-PHNet: Towards Real-Time and High-Precision LiDAR Panoptic Segmentation via Clustering Pseudo Heatmap. In IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- [27] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO: Common Objects in Context. In European Conference on Computer Vision, 2014.
- [28] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In *International Conference on Learning Representations*, 2017.
- [29] Rodrigo Marcuzzi, Lucas Nunes, Louis Wiesmann, Jens Behley, and Cyrill Stachniss. Mask-Based Panoptic LiDAR Segmentation for Autonomous Driving. In *IEEE Robotics and Automation Letters (RA-L)*, 2023.
- [30] Rodrigo Marcuzzi, Lucas Nunes, Louis Wiesmann, Elias Marks, Jens Behley, and Cyrill Stachniss. Mask4D: End-to-End Mask-based 4D Panoptic Segmentation for LiDAR Sequences. In *IEEE Robotics and Automation Letters (RA-L)*, 2023.
- [31] Rodrigo Marcuzzi, Lucas Nunes, Louis Wiesmann, Ignacio Vizzo, Jens Behley, and Cyrill Stachniss. Contrastive Instance Association for 4D Panoptic Segmentation using Sequences of 3D Lidar Scans. In *IEEE Robotics and Automation Letters (RA-L)*, 2022.
- [32] Andres Milioto, Ignacio Vizzo, Jens Behley, and C. Stachniss. RangeNet++: Fast and Accurate LiDAR Semantic Segmentation. In International Conference on Intelligent Robots and Systems, 2019.
- [33] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. In *International Conference on 3D Vision*, 2016.
- [34] Himangi Mittal, Brian Okorn, and David Held. Just Go With the Flow: Self-Supervised Scene Flow Estimation. In *IEEE Conference* on Computer Vision and Pattern Recognition, 2019.
- [35] Kenji Okuma, Ali Taleghani, Nando De Freitas, James J Little, and David G Lowe. A Boosted Particle Filter: Multitarget Detection and Tracking. In European Conference on Computer Vision, 2004.
- [36] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. In Neural Information Processing Systems, 2017.
- [37] Ryan Razani, Ran Cheng, Enxu Li, Ehsan Moeen Taghavi, Yuan Ren, and Bingbing Liu. GP-S3Net: Graph-based Panoptic Sparse Semantic Segmentation Network. In *International Conference on Computer Vision*, 2021.
- [38] Jonas Schult, Francis Engelmann, Alexander Hermans, Or Litany, Siyu Tang, and Bastian Leibe. Mask3D for 3D Semantic Instance Segmentation. In *IEEE International Conference on Robotics and Automation*, 2023.
- [39] Peer Schutt, Radu Alexandru Rosu, and Sven Behnke. Abstract Flow for Temporal Semantic Segmentation on the Permutohedral Lattice. In IEEE International Conference on Robotics and Automation, 2022.
- [40] Yichao Shen, Zigang Geng, Yuhui Yuan, Yutong Lin, Ze Liu, Chunyu Wang, Han Hu, Nanning Zheng, and Baining Guo. V-DETR: DETR with Vertex Relative Position Encoding for 3D Object Detection. In International Conference on Learning Representations, 2024.
- [41] Hanyu Shi, Guosheng Lin, Hao Wang, Tzu-Yi Hung, and Zhenhua Wang. SpSequenceNet: Semantic Segmentation Network on 4D Point Clouds. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2020.
- [42] Shaoshuai Shi, Chaoxu Guo, Li Jiang, Zhe Wang, Jianping Shi, Xiaogang Wang, and Hongsheng Li. PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2019.

- [43] Kshitij Sirohi, Rohit Mohan, Daniel Buscher, Wolfram Burgard, and Abhinav Valada. EfficientLPS: Efficient LiDAR Panoptic Segmentation. In *IEEE Transactions on Robotics*, 2021.
- [44] Leslie N. Smith and Nicholay Topin. Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates. In Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications, 2017.
- [45] Shihao Su, Jianyun Xu, Huanyu Wang, Zhenwei Miao, Xin Zhan, Dayang Hao, and Xi Li. PUPS: Point Cloud Unified Panoptic Segmentation. In Conference on Artificial Intelligence, 2023.
- [46] Jiahao Sun, Chunmei Qing, Junpeng Tan, and Xiangmin Xu. Superpoint Transformer for 3D Scene Instance Segmentation. In Conference on Artificial Intelligence, 2022.
- [47] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott M. Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in Perception for Autonomous Driving: Waymo Open Dataset. In IEEE Conference on Computer Vision and Pattern Recognition, 2019.
- [48] Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, and Ren Ng. Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. In Neural Information Processing Systems, 2020.
- [49] Haotian Tang, Zhijian Liu, Shengyu Zhao, Yujun Lin, Ji Lin, Hanrui Wang, and Song Han. Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution. In European Conference on Computer Vision, 2020.
- [50] Maxim Tatarchenko, Jaesik Park, Vladlen Koltun, and Qian-Yi Zhou. Tangent Convolutions for Dense Prediction in 3D. In IEEE Conference on Computer Vision and Pattern Recognition, 2018.
- [51] Hugues Thomas, Charles R Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, François Goulette, and Leonidas J Guibas. KPConv: Flexible and Deformable Convolution for Point Clouds. In *International Conference on Computer Vision*, 2019.
- [52] Paul Voigtlaender, Michael Krause, Aljosa Osep, Jonathon Luiten, Berin Balachandar Gnana Sekar, Andreas Geiger, and B. Leibe. MOTS: Multi-Object Tracking and Segmentation. In *IEEE Conference* on Computer Vision and Pattern Recognition, 2019.
- [53] Thang Vu, Kookhoi Kim, Tung M. Luu, Xuan Thanh Nguyen, and Chang D. Yoo. SoftGroup for 3D Instance Segmentation on 3D Point Clouds. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2022.
- [54] Song Wang, Jianke Zhu, and Ruixiang Zhang. Meta-RangeSeg: LiDAR Sequence Semantic Segmentation Using Multiple Feature Aggregation. In *IEEE Robotics and Automation Letters (RA-L)*, 2022.
- [55] Xinshuo Weng, Jianren Wang, David Held, and Kris Kitani. 3D Multi-Object Tracking: A Baseline and New Evaluation Metrics. In International Conference on Intelligent Robots and Systems, 2019.
- [56] Yan Yan, Yuxing Mao, and Bo Li. SECOND: Sparsely Embedded Convolutional Detection. In Sensors, 2018.
- [57] Linjie Yang, Yuchen Fan, and Ning Xu. Video Instance Segmentation. In International Conference on Computer Vision, 2019.
- [58] Tianwei Yin, Xingyi Zhou, and Philipp Krähenbühl. Center-based 3D Object Detection and Tracking. In IEEE Conference on Computer Vision and Pattern Recognition, 2020.
- [59] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene Parsing through ADE20K Dataset. In IEEE Conference on Computer Vision and Pattern Recognition, 2017.
- [60] Zixiang Zhou, Yang Zhang, and Hassan Foroosh. Panoptic-PolarNet: Proposal-free LiDAR Point Cloud Panoptic Segmentation. In IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- [61] Minghan Zhu, Shizong Han, Hong Cai, Shubhankar Borse, Maani Ghaffari Jadidi, and Fatih Porikli. 4D Panoptic Segmentation as Invariant and Equivariant Field Prediction. In *International Conference* on Computer Vision, 2023.
- [62] Xinge Zhu, Hui Zhou, Tai Wang, Fangzhou Hong, Yuexin Ma, Wei Li, Hongsheng Li, and Dahua Lin. Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2021.

# Mask4Former: Mask Transformer for 4D Panoptic Segmentation Supplementary Material

In this supplementary material, we demonstrate the versatility of Mask4Former by applying it to various segmentation tasks, showcasing its potential as a comprehensive 3D segmentation framework. Specifically, we use Mask4Former for both 3D panoptic segmentation and 4D semantic segmentation tasks. Our results indicate that Mask4Former achieves competitive performance across these tasks without any hyperparameter tuning or architectural modifications.

**3D panoptic segmentation** is the task of assigning a semantic class label for each point in a 3D scene while distinguishing different instances of the same class. Unlike 4D panoptic segmentation, which involves tracking instances over time, 3D panoptic segmentation processes each LiDAR scan independently. Transitioning from 4D to 3D panoptic segmentation for Mask4Former is straightforward by adjusting the number of superimposed LiDAR scans to 1. Evaluation is based on the panoptic quality metric, [21], calculated as follows:

$$PQ = \underbrace{\frac{\sum_{(p,g) \in TP} IoU(p,g)}{|TP|}}_{\text{segmentation quality (SQ)}} \times \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_{\text{recognition quality (RQ)}}$$
(3

For each class, prediction and ground truth masks are sorted into true positives, false positives, and false negatives. True positives represent pairs of prediction and ground truth masks with an IoU overlap of over 50%, false positives are unmatched predicted masks, and false negatives are unmatched ground truth masks. Segmentation quality is determined by the average IoU of matched segments, while recognition quality measures the F1 score, indicating the effectiveness of object recognition.

In Table V we report the scores on the SemanticKITTI 3D panoptic segmentation validation set. Our model achieves a PQ score of 61.7%, demonstrating competitive results compared to state-of-the-art techniques without any modifications. Notably, it outperforms the end-to-end trainable method [29] by 1.9%. Furthermore, our model achieves an SQ score of 80.8%, indicating that Mask4Former generates precise instance masks. However, a comparatively lower RQ score suggests that Mask4Former tends to produce many unmatched instance masks. This discrepancy might stem from the fact that the predicted number of instances  $(N_q)$  exceeds the actual number of instances in a LiDAR scene, which might result in a multitude of small mask predictions.

**4D semantic segmentation** is a semantic segmentation task where moving and stationary objects of the same category are treated as different semantic classes. As a result, there are 6 extra classes for moving objects such as "moving car" and "moving person" on top of the regular 19 classes. To distinguish between moving and stationary objects, the model needs to process multiple LiDAR scans together.

TABLE V: 3D panoptic segmentation scores on SemanticKITTI validation set.

| Method                 | PQ   | SQ   | RQ          |
|------------------------|------|------|-------------|
| DS-Net [18]            | 57.7 | 77.6 | 68.0        |
| Panoptic-PolarNet [60] | 59.1 | 78.3 | 70.2        |
| EfficientLPS [43]      | 59.2 | 75.0 | 69.8        |
| Mask-PLS [29]          | 59.8 | 76.3 | 69.0        |
| Panoptic-PHNet [26]    | 61.7 | -    | -           |
| GP-S3Net [37]          | 63.3 | 81.4 | 75.9        |
| PUPS [45]              | 64.4 | 81.5 | <u>74.1</u> |
| Mask4Former (Ours)     | 61.7 | 81.0 | 71.4        |

TABLE VI: 4D semantic segmentation scores on SemanticKITTI test set.

| Method                  | mIoU        | mIoU <sub>moving</sub> | mIoU <sub>static</sub> |
|-------------------------|-------------|------------------------|------------------------|
| TangentConv [50]        | 34.1        | 20.3                   | 38.5                   |
| DarkNet53Seg [50]       | 41.6        | 26.3                   | 46.4                   |
| SpSequenceNet [41]      | 43.1        | 26.5                   | 48.3                   |
| TemporalLidarSeg [13]   | 47.0        | 29.8                   | 52.4                   |
| TemporalLatticeNet [39] | 47.1        | 34.5                   | 51.1                   |
| Meta-RangeSeg [54]      | 49.7        | 38.1                   | 53.4                   |
| KPConv [51]             | 51.2        | 43.7                   | 53.6                   |
| Cylinder3D [62]         | 52.5        | 36.8                   | 57.5                   |
| MemorySeg [25]          | 58.3        | 53.4                   | <u>59.8</u>            |
| Mask4Former (Ours)      | <u>55.7</u> | 42.8                   | 59.8                   |

This is the reason why it is also referred to as multiscan semantic segmentation. Transitioning from 4D panoptic segmentation to 4D semantic segmentation requires two minor modifications. Firstly, instead of generating a target mask for each instance, a single target mask per class is generated. Consequently, the spatiotemporal queries predict all points belonging to a semantic class together. Secondly, bounding box parameter regression is omitted since a single target mask may encompass multiple instances of the same class. The evaluation metric for this task is mean Intersection over Union (mIoU) computed across 25 classes.

In Table VI we report the scores on the SemanticKITTI 4D semantic segmentation test set. Our results demonstrate notable achievements using the Mask4Former framework, with an mIoU of 55.7%, securing the second position on the leaderboard. We achieve an mIoU<sub>static</sub> of 59.8% on par with the state-of-the-art, showing we can accurately segment static objects. However, the mIoU<sub>moving</sub> of 42.8% suggests challenges in distinguishing between moving and static objects of the same class. This discrepancy may be attributed to our decision to superimpose only 2 consecutive LiDAR scans, as opposed to the 13 scans utilized in [25], to maintain consistency with the 4D panoptic segmentation framework. This may hinder our model's ability to capture sufficient temporal context for effectively distinguishing between these object categories.