Number of distinct tasks evaluated: 3. "We evaluate the effectiveness of the presented Point Transformer design on a number of domains and tasks. For 3D semantic segmentation, we use the challenging Stanford Large-Scale 3D Indoor Spaces (S3DIS) dataset [1]. For 3D shape classification, we use the widely adopted ModelNet40 dataset [47]. And for object part segmentation, we use ShapeNetPart [52]." (Section 4. Experiments)

Number of trained model instances required to cover all tasks: 3. "For semantic segmentation on S3DIS, we train for 40K iterations with initial learning rate 0.5, dropped by 10x at steps 24K and 32K. For 3D shape classification on ModelNet40 and 3D object part segmentation on ShapeNetPart, we train for 200 epochs." (Section 4. Experiments) "For semantic segmentation, the final decoder stage produces a feature vector for each point in the input point set. We apply an MLP to map this feature to the final logits. For classification, we perform global average pooling over the pointwise features to get a global feature vector for the whole point set. This global feature is passed through an MLP to get the global classification logits." (Section 3.5. Network Architecture — Output head)

Task–Model Ratio:

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
