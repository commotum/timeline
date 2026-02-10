1. **Number of distinct tasks evaluated:** 4

"After pre-training, we apply our model for zeroshot transfer *without* any fine-tuning on target dataset labels. We directly use our pre-trained model on a diverse set of *four* tasks in *five* datasets, including text-video retrieval (for text-to-video similarity), VideoQA (for video-to-text similarity), action localization (for video frame to text label similarity) and segmentation (for video token to text label similarity with rejection) (see §4)." (Section 1 Introduction)

2. **Number of trained model instances required to cover all tasks:** 1 model

"We present VideoCLIP, a contrastive approach to pre-train a unified model for zeroshot video and text understanding, without using any labels on downstream tasks." (Abstract)

"We present methods for zero-shot transfer of VideoCLIP to a variety of end tasks (*without* using any labels)." (Section 4 Zero-shot Transfer to End Tasks)

"In summary, the main contributions of this paper include: (i) we propose to pre-train a *unified* model that is capable of zero-shot transfer to *multiple* end tasks for video-text understanding..." (Section 1 Introduction)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{4\ \text{tasks}}{1\ \text{model}} = 4
}
$$
