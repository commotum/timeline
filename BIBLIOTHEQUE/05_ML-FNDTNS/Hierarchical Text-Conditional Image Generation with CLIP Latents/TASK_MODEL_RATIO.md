1. **Number of distinct tasks evaluated:** 4
   - "#### 5 Text-to-Image Generation" (Section 5)
   - "We describe three different kinds of manipulations that are enabled by this bipartite representation." (Section 3, *Image Manipulations*)
   - "#### 3.1 Variations"; "#### 3.2 Interpolations"; "#### 3.3 Text Diffs" (Sections 3.1-3.3)

2. **Number of trained model instances required to cover all tasks:** 2 models
   - "we propose a two-stage model: a prior that generates a CLIP image embedding given a text caption, and a decoder that generates an image conditioned on the image embedding." (Abstract)
   - "We design our generative stack to produce images from captions using two components:" followed by "A prior  $P(z_i|y)$" and "A decoder  $P(x|z_i,y)$" (Section 2, *Method*)

3. **Task–Model Ratio**

$$
\boxed{
\frac{4\ \text{tasks}}{2\ \text{models}} = 2
}
$$
