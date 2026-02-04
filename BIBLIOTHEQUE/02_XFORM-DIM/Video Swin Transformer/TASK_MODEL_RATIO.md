Number of distinct tasks evaluated: 3.
Quote (Section 1 Introduction): "The proposed approach shows strong performance on the video recognition tasks of action recognition on Kinetics-400/Kinetics-600 and temporal modeling on Something-Something v2 (abbreviated as SSv2)."
Quote (Section 4.1 Setup): "For human action recognition, we adopt two versions of the widely-used Kinetics [20] dataset, Kinetics-400 and Kinetics-600."
Quote (Section 4.1 Setup): "For temporal modeling, we utilize the popular Something-Something V2 (SSv2) [14] dataset, which consists of 168.9K training videos and 24.7K validation videos over 174 classes."

Number of trained model instances required to cover all tasks: 3.
Quote (Section 4.1 Setup): "For K400 and K600, we employ an AdamW [21] optimizer for 30 epochs using a cosine decay learning rate scheduler and 2.5 epochs of linear warm-up."
Quote (Section 4.1 Setup): "For SSv2, we employ an AdamW [21] optimizer for longer training of 60 epochs with 2.5 epochs of linear warm-up."
Quote (Section 4.2 Comparison to state-of-the-art, Something-Something v2): "We follow MViT [9] by using the K400 pre-trained model as initialization and a window size in temporal dimension of 16 is used."

Task-Model Ratio = (1) / (2): 1.

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
