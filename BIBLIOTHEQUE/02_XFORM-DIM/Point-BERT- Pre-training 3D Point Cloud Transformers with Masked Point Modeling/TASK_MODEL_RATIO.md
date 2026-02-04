1. Number of distinct tasks evaluated: 4. Quote (Section 4. Experiments): "Then we evaluate the proposed model with various downstream tasks, including object classification, part segmentation, few-shot learning and transfer learning."
2. Number of trained model instances required to cover all tasks: 4. Quotes (Section 4.2. Downstream Tasks): "In the classification task, a two-layer MLP with a dropout of 0.5 is used as our classification head." "We design a segmentation head to propagate the group features to each point hierarchically." "The model is trained on  $K \times N$  samples (support set), and evaluated on the remaining 20K samples (query set)." "We evaluate the generalization ability of the learned representation by pre-training the model on ShapeNet and fine-tuning it on ScanObjectNN [49], which contains 2902 point clouds from 15 categories."
3. Task–Model Ratio:
$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
