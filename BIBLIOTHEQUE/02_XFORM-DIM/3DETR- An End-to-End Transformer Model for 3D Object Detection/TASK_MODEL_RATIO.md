Section 4. Experiments: "We evaluate models on two standard 3D indoor detection benchmarks - ScanNetV2 [7] and SUN RGB-D-v1 [53]."

Section 4.2 (Encoder applied to Shape classification): "To verify that our encoder design is not specific to the detection task we test the encoder on shape classification of of models including 3D Warehouse [79]."

Table 4: "Table 4: Shape classification. We report shape classification results by training our Transformer encoder model."

Number of distinct tasks evaluated: 2 (3D object detection; shape classification).

Number of trained model instances required to cover all tasks: 2 (3D object detection model; separately trained Transformer encoder + MLP for shape classification).

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
