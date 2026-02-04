1. Number of distinct tasks evaluated: 3.
Quote: "RoPE in ViT and Swin Transformer is validated for image recognition, including multi-resolution classification (§4.1) on ImageNet-1k [4], object detection (§4.2) on MS-COCO [16], and semantic segmentation (§4.3) on ADE20k [40,41]." (Section 4 Experiments)

2. Number of trained model instances required to cover all tasks: 3.
Quote: "We train ViTs and Swin Transformers on ImageNet-1k [4] training set with high-performance training recipes [17,32]." (Section 4.1 Multi-resolution classification)
Quote: "DINO [39] detector is trained using ViT and Swin as backbone network." (Section 4.2 Object detection)
Quote: "We train 2D RoPE ViT and Swin for semantic segmentation on ADE20k [40, 41]. For ViT, we use UperNet [37] with ViT training recipe [21]. For Swin, Mask2Former [2] for segmentation is used with the Swin." (Section 4.3 Semantic segmentation)

3. Task–Model Ratio = 1.

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
