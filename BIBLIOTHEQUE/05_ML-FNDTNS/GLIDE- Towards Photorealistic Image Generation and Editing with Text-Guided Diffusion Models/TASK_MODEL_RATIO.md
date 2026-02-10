1. **Number of distinct tasks evaluated:** 2

- "we explore diffusion models for the problem of text-conditional image synthesis" (Abstract)
- "our models can be fine-tuned to perform image inpainting, enabling powerful text-driven image editing." (Abstract)

2. **Number of trained model instances required to cover all tasks:** 4

- "For our main experiments, we train a 3.5 billion parameter text-conditional diffusion model at  $64 \times 64$  resolution, and another 1.5 billion parameter text-conditional upsampling diffusion model to increase the resolution to  $256 \times 256$ ." (Section 4. Training)
- "To achieve better results, we explicitly fine-tune our model to perform inpainting" (Section 4.3. Image Inpainting)
- "For the upsampling model, we always provide the full low-resolution image, but only provide the unmasked region of the high-resolution image." (Section 4.3. Image Inpainting)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{2\ \text{tasks}}{4\ \text{models}} = 0.5
}
$$
