> "We compare RoMAE with state-of-the-art deep learning (DL) models, conducting experiments on the following tasks and modalities: (i) irregularly sampled multi-variate time-series classification, (ii) image classification, (iii) irregularly sampled time-series interpolation and (iv) audio classification." (Section 1 Introduction)

> "To verify the model's ability to reconstruct absolute positional information according to Proposition 4.2, we give the model a sequence of 10 identical values as input. Each embedding is then given a 1D position sampled uniformly between 0 and 50. We then use the same linear head to predict the position for all tokens." (Section 5.1 Reconstructing Absolute Position)

> "RoMAE is trained directly on regression without any pre-training, predicting the sine and cosine of the angle of the pendulum which follows a non-linear dynamical system." (Section 5.4 Irregular Time-series Regression: Pendulum Dataset)

> "When fine-tuning RoMAE without the [CLS] token, we place the classification head on top of the mean of the output embeddings, otherwise we place the head on top of the [CLS] token." (Section 5.2 Tiny ImageNet)

Number of distinct tasks evaluated: 6.

Number of trained model instances required to cover all tasks: 6.

$$
\boxed{
\frac{6\ \text{tasks}}{6\ \text{models}} = 1
}
$$
