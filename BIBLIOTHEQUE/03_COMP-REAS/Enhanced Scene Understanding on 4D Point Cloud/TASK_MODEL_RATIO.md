1. **Number of distinct tasks evaluated:** 3

   Verbatim evidence:
   - Section `Problem Formulation`: "Typically, there are three main tasks in the 4D point cloud analysis: 4D semantic segmentation, action segmentation and action recognition."
   - Section `Experiments Setup` (`Datasets`): "The above datasets include three tasks: 4D action segmentation, 4D action recognition and 4D semantic segmentation."

2. **Number of trained model instances required to cover all tasks:** 3

   Verbatim evidence:
   - Section `Cross-modal Transformer`: "Finally, the output feature is utilized in several 4D task heads for downstream tasks, such as 4D action segmentation."
   - Section `Problem Formulation`: "SemSeg: \mathbb{R}^{T \times N \times 3} \mapsto \mathbb{R}^{T \times N},", "ActionSeg: \mathbb{R}^{T \times N \times 3} \mapsto \mathbb{R}^{T},", and "ActionRecog: \mathbb{R}^{T \times N \times 3} \mapsto \mathbb{R}^1".
   - Whether all tasks are handled by one jointly trained unified model instance without task-specific heads: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
