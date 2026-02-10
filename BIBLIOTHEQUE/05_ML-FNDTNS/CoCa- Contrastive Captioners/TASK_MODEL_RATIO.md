1. **Number of distinct tasks evaluated:** **8**
   - "Our visual recognition experiments are conducted on ImageNet [9] as image recognition benchmark, and multiple video datasets including Kinetics-400 [57], Kinetics-600 [58], Kinetics-700 [59], Moments-in-Time [60] as test-beds for video action recognition;" (Section 4.2.1)
   - "**Zero-Shot Image-Text Retrieval.**" (Section 4.2.2)
   - "**Zero-Shot Image Classification.**" (Section 4.2.2)
   - "**Zero-Shot Video Retrieval.**" (Section 4.2.2)
   - "We consider three popular multimodal understaning benchmarks: visual question answering (VQA v2 [75]), visual entailment (SNLI-VE [76]), and visual reasoning (NLVR2 [77])." (Section 4.2.3)
   - "**Image Captioning.**" (Section 4.2.3)

2. **Number of trained model instances required to cover all tasks:** **6**
   - "CoCa sets new state-of-the-art results on tasks of all three categories with a single pretrained checkpoint." (Section 4.2)
   - "A pretrained CoCa model performs many tasks in a zero-shot manner by leveraging both image and text inputs, including zero-shot image classification, zero-shot image-text cross-retrieval, zero-shot video-text cross-retrieval." (Section 3.3)
   - "For frozenfeature evaluation or finetuning, we learn an additional pooler on top of the spatial and temporal feature tokens with a softmax cross-entropy loss." (Section 3.3, CoCa for Video Action Recognition)
   - "We mainly follow the settings in [16] and train linear classifiers on top of the decoder outputs to predict answers" and "We consider three popular multimodal understaning benchmarks: visual question answering (VQA v2 [75]), visual entailment (SNLI-VE [76]), and visual reasoning (NLVR2 [77])." (Section 4.2.3)
   - "We finetune CoCa with the captioning loss  $\mathcal{L}_{Cap}$  only on MSCOCO [63] captioning task" (Section 4.2.3)

3. **Task-Model Ratio = (1) / (2)**

$$
\boxed{
\frac{8\ \text{tasks}}{6\ \text{models}} = 1.33
}
$$
