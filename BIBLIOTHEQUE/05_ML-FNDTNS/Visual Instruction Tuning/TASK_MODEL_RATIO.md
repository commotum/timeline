1. **Number of distinct tasks evaluated:** 4

- Quote (Section 5.1, LLaVA-Bench (COCO)): "For each image, we generate three types of questions (conversation, detailed description, complex reasoning) using the proposed data generation pipeline in Sec. 3, totaling 90 questions."
- Quote (Section 4.2, Stage 2: Fine-tuning End-to-End): "Science QA. We study our method on the ScienceQA benchmark [34], the first large-scale multimodal science question dataset that annotates the answers with detailed lectures and explanations."

2. **Number of trained model instances required to cover all tasks:** 2

- Quote (Section 4.2 Training): "We consider two specific use case scenarios:"
- Quote (Section 4.2, Stage 2: Fine-tuning End-to-End): "Multimodal Chatbot. We develop a Chatbot by fine-tuning on the 158K language-image instruction-following data in Section 3."
- Quote (Section 4.2, Stage 2: Fine-tuning End-to-End): "Science QA. We study our method on the ScienceQA benchmark [34], the first large-scale multimodal science question dataset that annotates the answers with detailed lectures and explanations."
- Quote (Section 5 Experiments): "We assess the performance of LLaVA in instruction-following and visual reasoning capabilities with two primary experimental settings: multimodal chatbot and the ScienceQA dataset, respectively."

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{4\ \text{tasks}}{2\ \text{models}} = 2
}
$$
