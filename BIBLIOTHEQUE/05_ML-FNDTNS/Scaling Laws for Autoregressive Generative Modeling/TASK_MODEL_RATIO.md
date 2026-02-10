1. **Number of distinct tasks evaluated:** 6

- "We identify empirical scaling laws for the cross-entropy loss in four domains: generative image modeling, video modeling, multimodal image text models, and mathematical problem solving." (Abstract)
- "We separately study models for text-to-image and image-to-text mappings, as we found poor performance for bidirectional models in preliminary experiments." (Section 2.1.5: Multimodal Text and Images)
- "we finetune generative image models for ImageNet classification and find smooth scaling of the classification loss and error rate, even as the generative loss levels off." (Abstract)
- "We show results from GPT-3 [BMR<sup>+</sup>20] for comparison" (Section 2.1.1: Language)

2. **Number of trained model instances required to cover all tasks:** 6

- "Moreover, we demonstrate that a single architecture – the Transformer [VSP<sup>+</sup>17, LSP<sup>+</sup>18], with an autoregressive cross-entropy loss – scales smoothly in all of these domains, with only minimal changes to hyperparameters such as width, depth, or learning rate." (Section 1: Introduction)
- "We separately study models for text-to-image and image-to-text mappings, as we found poor performance for bidirectional models in preliminary experiments." (Section 2.1.5: Multimodal Text and Images)
- "To turn these models into classifiers, we remove their final embedding matrix and use the mean-pooled (over all pixels) activations of the transformer's final layer as the input to a new single-layer classifier." (Section 3.4: Finetuning on ImageNet at 32x32 Resolution)
- "During finetuning we backpropagate through the full transformer, and we do not freeze any of its weights." (Section 3.4: Finetuning on ImageNet at 32x32 Resolution)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{6\ \text{tasks}}{6\ \text{models}} = 1
}
$$
