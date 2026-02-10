1. **Number of distinct tasks evaluated: 3**
- "BLIP-2 achieves state-of-the-art performance on various vision-language tasks including visual question answering, image captioning, and image-text retrieval." (Section 1. Introduction)
- "Table 1. Overview of BLIP-2 results on various **zero-shot** vision-language tasks." (Table 1)

2. **Number of trained model instances required to cover all tasks: 3**
- "We finetune BLIP-2 models for the image captioning task, which asks the model to generate a text description for the image's visual content." (Section 4.2. Image Captioning)
- "Given annotated VQA data, we finetune the parameters of the Q-Former and the image encoder while keeping the LLM frozen." (Section 4.3. Visual Question Answering)
- "Since image-text retrieval does not involve language generation, we directly finetune the first-stage-pretrained model w/o LLM." (Section 4.4. Image-Text Retrieval)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
