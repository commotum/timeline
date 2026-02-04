> ## 5.1 Linear SVM Classification
> Linear SVM classification task has become a classic task to evaluate self-supervised point cloud representation learning.
> We used our pre-training model to extract the features of each point cloud, then trained a simple linear Support Vector Machine (SVM) on the training set of ModelNet40, and finally tested the SVM on the ModelNet40 test set.

> #### 5.2 Downstream Tasks
> **3D Object Classification on Synthetic Data** To test whether POS-BERT can help boost downstream tasks. We first performed fine-tuning experiments on point cloud classification tasks using a pretraining model.
> **Few-shot Classification** To demonstrate that our pre-training model can learn quickly from few-shot samples, we conduct experiment on the Few-shot ModelNet40 dataset.
> **3D Object Classification on Real-world Data** In this experiment, we aim to explore whether the knowledge POS-BERT learns from ShapNet can be transferred to real-world data. We conduct experiments on three variants of ScanObjectNN [60] dataset, including OBJ-BG, OBJ-ONLY, and PB-T50-RS.
> **Part Segmentation** In this section, we explore how the pre-training model performs in the pre-point classification. We experimented on ShapeNetPart, a benchmark dataset commonly used in point cloud segmentation tasks.

> ## 4.1 Implementation
> **Classification** We use a fully connected MLP network that combines ReLU, BN, and Dropout operations as the classification head.
> **Segmentation** Different from the classification task, the segmentation task needs to predict pre-point labels.
> Finally, MLP is used to map the features to the segmentation label space.

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$
