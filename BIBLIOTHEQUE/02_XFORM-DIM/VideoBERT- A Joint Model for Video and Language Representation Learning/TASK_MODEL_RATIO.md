Number of distinct tasks evaluated: 2
Section 3.2:
> Once we have trained the model, we can use it in a variety of downstream tasks, and in this work we quantitatively evaluate two applications.
> We illustrate this in Section 4.4, where we perform "zero-shot" classification.
> We demonstrate this in Section 4.6, where we perform video captioning.

Number of trained model instances required to cover all tasks: 2
Section 4.3:
> Our model training process largely follows the setup of BERT: we use 4 Cloud TPUs in the Pod configuration with a total batch size of 128, and we train the model for 0.5 million iterations, or roughly 8 epochs.
Section 4.6:
> We evaluate the extracted features on video captioning, following the setup from [39], where the ground truth video segmentations are used to train a supervised model mapping video segments to captions.
> We train the model for 40K iterations with batch size of 128.

Task–Model Ratio:
$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
