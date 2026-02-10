1. **Number of distinct tasks evaluated:** 1

   - "We propose a new task of streaming 4D panoptic segmentation." (Section 3: Streaming 4D Panoptic Segmentation)
   - "Our goal is to develop an approach that finds a trade-off between accuracy and efficiency to enable real-time inference for the Streaming 4D Panoptic Segmentation task." (Section 3: Streaming 4D Panoptic Segmentation)

2. **Number of trained model instances required to cover all tasks:** 1

   - "This system consists of a Predictive Thread for memory updating and future dynamics forecasting and an Inference Thread that allows incoming future points to quickly retrieve the corresponding features from memory, ensuring efficient inference within the limited time constraints." (Section 4.1: Dual-thread system)
   - "However, unlike traditional fast-slow systems that rely on separate models for fast and slow tasks, 4DSegStreamer integrates both components into a unified pipeline, enabling seamless interaction between memory updates and real-time queries." (Section 2.3: Fast-slow Dual System Methods)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{1\ \text{task}}{1\ \text{model}} = 1
}
$$
