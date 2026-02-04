1. Number of distinct tasks evaluated: 4. "Extensive experiments demonstrate that the PCT achieves the state-of-the-art performance on shape classification, part segmentation, semantic segmentation and normal estimation tasks." (Abstract)
2. Number of trained model instances required to cover all tasks: 4. "We use different architectures for the tasks of point cloud classification, segmentation and normal estimation." (Section 3.4. Neighbor Embedding for Augmented Local Feature Representation) "For the task of segmenting the point cloud into  $N_s$  parts (e.g. table top, table legs; a part need not be contiguous)" (Section 3.1. Point Cloud Processing with PCT) "It has 16 object categories and 50 part labels;" (Section 4.3. Segmentation task on ShapeNet dataset) "Each point in the dataset is divided into 13 categories." (Section 4.4. Semantic segmentation task on S3DIS dataset) "For the task of normal estimation, we use the same architecture as in segmentation by setting  $N_s=3$ , without the object category encoding, and regard the output point-wise score as the predict normal." (Section 3.1. Point Cloud Processing with PCT)
3. Task-Model Ratio = (1) / (2):

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
