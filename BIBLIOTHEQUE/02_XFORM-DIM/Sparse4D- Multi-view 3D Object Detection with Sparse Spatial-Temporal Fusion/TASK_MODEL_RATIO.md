1. Number of distinct tasks evaluated: 2. Quote (Section 4.1. Datasets and Metrics): "For the 3D detection task, evaluation metrics include mean Average Precision (mAP), mean Average Error of Translation (mATE), Scale (mASE), Orientation (mAOE), Velocity (mAVE), Attribute (mAAE) and nuScenes Detection Score (NDS), where NDS is a weighted average of other metrics. For the object tracking task, Average Multi-Object Tracking Accuracy (AMOTA), Average Multi-Object Tracking Precision (AMOTP) and Recall are the three main evaluation metrics."
2. Number of trained model instances required to cover all tasks: 2. Quote (Section 4.5. Extend to 3D Object Tracking): "Based on the tracking-by-detection framework [6], Sparse4D is easily extended to a tracker. We use the instance features and bounding boxes output by the last refinement module to extract identity features, and use a lightweight sub-network to estimate the correlation matrix between historical trajectories and current objects."

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
