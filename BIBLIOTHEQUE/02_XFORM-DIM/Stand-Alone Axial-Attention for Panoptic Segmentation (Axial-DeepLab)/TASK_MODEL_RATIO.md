1 Introduction: "We show the effectiveness of our axial-attention models on ImageNet [70] for classification, and on three datasets (COCO [56], Mapillary Vistas [62], and Cityscapes [22]) for panoptic segmentation [45], instance segmentation, and semantic segmentation."
4 Experimental Results: "We first report results with our Axial-ResNet on ImageNet [70]. We then convert the ImageNet pretrained Axial-ResNet to Axial-DeepLab, and report results on COCO [56], Mapillary Vistas [62], and Cityscapes [22] for panoptic segmentation, evaluated by panoptic quality (PQ) [45]. We also report average precision (AP) for instance segmentation, and mean IoU for semantic segmentation on Mapillary Vistas and Cityscapes."
3.2 Axial-Attention (Axial-DeepLab): "The heads produce semantic segmentation and class-agnostic instance segmentation, and they are merged by majority voting [89] to form the final panoptic segmentation."
Number of distinct tasks evaluated: 4.
Number of trained model instances required to cover all tasks: 2.

$$
\boxed{
\frac{4\ \text{tasks}}{2\ \text{models}} = 2
}
$$
