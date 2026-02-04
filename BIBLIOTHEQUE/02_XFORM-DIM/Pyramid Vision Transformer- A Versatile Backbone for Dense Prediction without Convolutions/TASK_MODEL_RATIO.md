1. Number of distinct tasks evaluated: 4. Quote (Section 1. Introduction): "- (3) We evaluate the proposed PVT on several different tasks, including image classification, object detection, instance and semantic segmentation, and compare it with popular ResNets [22] and ResNeXts [73]."
2. Number of trained model instances required to cover all tasks: 4. Quote (Section 4.1. Image-Level Prediction): "For image classification, we follow ViT [13] and DeiT [63] to append a learnable classification token to the input of the last stage, and then employ a fully connected (FC) layer to conduct classification on top of the token." Quote (Section 4.2. Pixel-Level Dense Prediction): "We apply our PVT models to three representative dense prediction methods, namely RetinaNet [39], Mask R-CNN [21], and Semantic FPN [32]. RetinaNet is a widely used single-stage detector, Mask R-CNN is the most popular two-stage instance segmentation framework, and Semantic FPN is a vanilla semantic segmentation method without special operations (*e.g.*, dilated convolution)."
3. Task-Model Ratio:
$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
