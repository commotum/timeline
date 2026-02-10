1. **Number of distinct tasks evaluated:** 8

   "We evaluate BEIT-3 on extensive downstream tasks and datasets, i.e., object detection (COCO), instance segmentation (COCO), semantic segmentation (ADE20K), image classification (ImageNet), visual reasoning (NLVR2), visual question answering (VQAv2), image captioning (COCO), and cross-modal retrieval (Flickr30K, COCO)." (Section 1: Introduction: The Big Convergence)

2. **Number of trained model instances required to cover all tasks:** 7

   "BEIT-3 is finetuned as a fusion encoder to model deep interactions of images and questions for the VQA task." and "We finetune BEIT-3 as a fusion encoder to jointly encode the image-text pairs." (Section 3.1: Vision-Language Downstream Tasks)

   "BEIT-3 is used as a conditional generation model via masked finetuning." (Section 3.1: Vision-Language Downstream Tasks)

   "BEIT-3 is finetuned as a dual encoder for efficient image-text retrieval." (Section 3.1: Vision-Language Downstream Tasks)

   "Table 7: Results of object detection and instance segmentation on COCO benchmark. BEIT-3 uses Cascade Mask R-CNN [CV21] as the detection head." (Table 7, Section 3.2: Vision Downstream Tasks)

   "We use a dense prediction task adapter and employ Mask2Former [CMS+21] as the segmentation framework." (Section 3.2: Vision Downstream Tasks)

   "BEIT-3 is trained as a dual encoder to find the most relevant label for an image." (Section 3.2: Vision Downstream Tasks)

   "Table 12: Hyperparameters for fine-tuning BEIT-3 on NLVR2 and VQAv2."; "Table 13: Hyperparameters for fine-tuning BEIT-3 on COCO captioning."; "Table 14: Hyperparameters for fine-tuning BE<sub>1</sub>T-3 on image-text retrieval."; "Table 15: Hyperparameters for fine-tuning BEIT-3 on semantic segmentation."; "Table 16: Hyperparameters for fine-tuning BEIT-3 on object detection."; "Table 17: Hyperparameters for fine-tuning BEIT-3 on image classification." (Appendix C: Hyperparameters Used for Finetuning)

   A single jointly fine-tuned model instance that covers all downstream tasks: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{8\ \text{tasks}}{7\ \text{models}} = \frac{8}{7}
}
$$
