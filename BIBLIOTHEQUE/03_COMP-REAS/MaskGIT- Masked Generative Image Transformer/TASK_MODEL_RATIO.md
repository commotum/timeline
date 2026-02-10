1. **Number of distinct tasks evaluated:** 4

   "In 4.2, we evaluate MaskGIT on the standard class-conditional image generation tasks on ImageNet [10] 256×256 and 512×512." (Section 4. Experiments)

   "In 4.3, we show MaskGIT's versatility by demonstrating its performance on three image editing tasks, image inpainting, outpainting, and editing." (Section 4. Experiments)

2. **Number of trained model instances required to cover all tasks:** 2

   "For each dataset, we only train a single autoencoder, decoder, and codebook with 1024 tokens on cropped 256x256 images for all the experiments." (Section 4.1. Experimental Setup)

   "ImageNet models are trained for 300 epochs while the Places2 model is trained for 200 epochs." (Section 4.1. Experimental Setup)

   "To match the training of our baselines, we train MaskGIT on the 512×512 center-cropped images from the Places2 [58] dataset." (Section 4.3. Image Editing Applications, Image Inpainting)

3. **Task–Model Ratio**

$$
\boxed{
\frac{4\ \text{tasks}}{2\ \text{models}} = 2
}
$$
