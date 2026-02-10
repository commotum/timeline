1. **Number of distinct tasks evaluated:** 3

   "We perform a number of experiments on a few different networks and three datasets:" followed by "For the MNIST dataset," "The ImageNet dataset [3]." and "~10 M image samples from Youtube (see [10])" (Section 2, Framework). The third setting is explicitly task-defined as: "A binary car classifier was trained on top of the last layer features without fine-tuning." (Figure 6 caption, Section 4.2).

2. **Number of trained model instances required to cover all tasks:** 3

   The paper uses separate trained systems for these task settings: "A simple fully connected network with one or more hidden layers and a Softmax classifier" / "A classifier trained on top of an autoencoder" (MNIST), "Krizhevsky et. al architecture [9]. We refer to it as \"AlexNet\"" (ImageNet), and "A binary car classifier was trained on top of the last layer features without fine-tuning." (QuocNet setting) (Section 2, Figure 6 in Section 4.2).

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
