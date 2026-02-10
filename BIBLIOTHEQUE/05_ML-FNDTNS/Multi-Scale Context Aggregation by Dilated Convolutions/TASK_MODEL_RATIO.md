1. **Number of distinct tasks evaluated:** 1 (semantic segmentation)

   - “State-of-the-art models for semantic segmentation are based on adaptations of convolutional networks that had originally been designed for image classification.” (ABSTRACT)
   - “Table 3 shows the effect of adding the context module to three different architectures for semantic segmentation.” (Section 5 EXPERIMENTS)
   - “Table 5: Semantic segmentation results on the CamVid dataset.” (Appendix A.1 CAMVID)
   - “Table 6: Semantic segmentation results on the KITTI dataset.” (Appendix A.2 KITTI)

2. **Number of trained model instances required to cover all tasks:** 1 model

   - “The first architecture (top) is the front end described in Section 4. It performs semantic segmentation without structured prediction, akin to the original work of Long et al. (2015).” (Section 5 EXPERIMENTS)
   - “The second architecture (Table 3, middle) uses the dense CRF to perform structured prediction, akin to the system of Chen et al. (2015a).” (Section 5 EXPERIMENTS)
   - “The third architecture (Table 3, bottom) uses the CRF-RNN for structured prediction (Zheng et al., 2015).” (Section 5 EXPERIMENTS)
   - “They were obtained with convolutional networks that combine a front-end module and a context module, akin to the "Front + Basic" network evaluated in Table 3.” (APPENDIX A URBAN SCENE UNDERSTANDING)

3. **Task–Model Ratio**

$$
\boxed{
\frac{1\ \text{task}}{1\ \text{model}} = 1
}
$$
