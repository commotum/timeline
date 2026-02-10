1. **Number of distinct tasks evaluated:** 4

- "In this work, we propose a model, called SegPoint, that leverages the reasoning capabilities of a multi-modal Large Language Model (LLM) to produce point-wise segmentation masks across a diverse range of tasks: 1) 3D instruction segmentation, 2) 3D referring segmentation, 3) 3D semantic segmentation, and 4) 3D open-vocabulary semantic segmentation." (Abstract)
- "Taking advantage of a multi-modal LLM and task-specific prompts, SegPoint is capable of generating segmentation masks for a wide range of tasks in a unified model: 1) 3D instruction segmentation, 2) 3D referring segmentation, 3) 3D semantic segmentation, and 4) 3D open-vocabulary semantic segmentation, as depicted in Fig. 1." (§1 Introduction)

2. **Number of trained model instances required to cover all tasks:** 4 models

- "During training, we use all mentioned datasets in Sec. 4.1 for joint training by leveraging task-specific prompts. For evaluation on a specific dataset, we finetune the trained model on the corresponding dataset." (§4.2 Implementation Details)
- "To ensure fair comparisons, we fine-tune our model on each semantic segmentation dataset to accommodate the varying class category definitions." (§4.4 Results on Semantic Segmentation)
- "This benchmark incorporates 280 scenes specifically selected for instruction segmentation tuning and evaluation, sourced from the recently introduced ScanNet++ [74] dataset." (§3.6 Instruct3D Dataset Collection)
- Not specified in the paper: a single post-training checkpoint evaluated across all four tasks without task-specific fine-tuning.

3. **Task–Model Ratio**

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
