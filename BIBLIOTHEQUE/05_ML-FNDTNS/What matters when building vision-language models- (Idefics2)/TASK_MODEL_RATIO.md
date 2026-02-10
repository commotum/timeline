1. **Number of distinct tasks evaluated:** 8 tasks.

   "Unless specified otherwise, we run the ablations for 6'000 steps and report the average score of the 4-shot performance on 4 downstream benchmarks measuring different capabilities: VQAv2 (Goyal et al., 2017) for general visual question answering, TextVQA (Singh et al., 2019) for OCR abilities, OKVQA (Marino et al., 2019) for external knowledge, and COCO (Lin et al., 2014) for captioning." (Section 3)

   "We evaluate Idefics2 on commonly adopted benchmarks: MMMU (Yue et al., 2024) for multidiscipline college-level problems, MathVista (Lu et al., 2024) for mathematical reasoning, TextVQA" and "(Singh et al., 2019) for text reading on natural images, and MMBench Liu et al. (2023) for various perception and reasoning tasks." (Section 4.2)

   "For the open-ended questions in TextVQA, DocVQA, and VQAv2, we evaluate with the prompt:" (Section A.3.1)

2. **Number of trained model instances required to cover all tasks:** 2 models.

   "To evaluate the base model, we consider VQAv2 (Goyal et al., 2017), TextVQA (Singh et al., 2019), OKVQA (Marino et al., 2019), and COCO (Lin et al., 2014)." (Section 4.1)

   "We continue the training with an instruction fine-tuning phase." (Section 4.2)

   "Idefics2 with 64 or 320 tokens per image is the same model (same weights), only the inference differs." (Table 9, Section 4.2)

3. **Task–Model Ratio**

$$
\boxed{
\frac{8\ \text{tasks}}{2\ \text{models}} = 4
}
$$
