1. **Number of distinct tasks evaluated:** 7

"(ii) We achieve competitive performance on multiple tasks (unconditional image synthesis, inpainting, stochastic super-resolution) and datasets while significantly lowering computational costs." (Section 1, Contributions)

"(iv) We find that for densely conditioned tasks such as super-resolution, inpainting and semantic synthesis, our model can be applied in a convolutional fashion and render large, consistent images of  $\sim 1024^2$  px." (Section 1, Contributions)

"(v) Moreover, we design a general-purpose conditioning mechanism based on cross-attention, enabling multi-modal training. We use it to train class-conditional, text-to-image and layout-to-image models." (Section 1, Contributions)

2. **Number of trained model instances required to cover all tasks:** 7

"A notable advantage of this approach is that we need to train the universal autoencoding stage only once and can therefore reuse it for multiple DM trainings or to explore possibly completely different tasks [81]." (Section 1, Introduction)

"Table 12. Hyperparameters for the unconditional *LDMs* producing the numbers shown in Tab. 1. All models trained on a single NVIDIA A100." (Section E.1, Table 12)

"| Task                | Text-to-Image           | Layout-to-Image         |                         | Class-Label-to-Image    | Super Resolution        | Inpainting              | Semantic-Map-to-Image   |  |" and "Table 15. Hyperparameters for the conditional *LDMs* from Sec. 4. All models trained on a single NVIDIA A100 except for the inpainting model which was trained on eight V100." (Section E.1, Table 15)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{7\ \text{tasks}}{7\ \text{models}} = 1
}
$$
