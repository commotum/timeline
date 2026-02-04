Number of distinct tasks evaluated: 2. Evidence: "Text-to-image generation has traditionally focused on finding better modeling assumptions for training on a fixed dataset." (Abstract) Evidence: "To a limited degree of reliability, we also find our model to be capable of zero-shot image-to-image translation controllable by natural language (Figure 2d)." (Section 3.3. Qualitative Findings)

Number of trained model instances required to cover all tasks: 2. Evidence: "In the first stage of training, we maximize the ELB with respect to  $\phi$  and  $\theta$ , which corresponds to training a dVAE on the images alone." (Section 2.1. Stage One: Learning the Visual Codebook) Evidence: "In the second stage, we fix  $\phi$  and  $\theta$ , and learn the prior distribution over the text and image tokens by maximizing the ELB with respect to  $\psi$ . Here,  $p_{\psi}$  is represented by a 12-billion parameter sparse transformer (Child et al., 2019)." (Section 2.2. Stage Two: Learning the Prior)

Task-Model Ratio = 2 / 2 = 1.

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
