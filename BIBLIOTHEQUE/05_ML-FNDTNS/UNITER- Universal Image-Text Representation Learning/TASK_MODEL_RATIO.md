1. **Number of distinct tasks evaluated:** 6

   "Extensive experiments show that UNITER achieves new state of the art across six V+L tasks (over nine datasets), including Visual Question Answering, Image-Text Retrieval, Referring Expression Comprehension, Visual Commonsense Reasoning, Visual Entailment, and NLVR<sup>2</sup>." (Abstract)

   "We evaluate UNITER on six V+L tasks  $^{11}$  by transferring the pre-trained model to each target task and finetuning through end-to-end training." (Section 4: Experiments)

2. **Number of trained model instances required to cover all tasks:** 6

   "We evaluate UNITER on six V+L tasks  $^{11}$  by transferring the pre-trained model to each target task and finetuning through end-to-end training." (Section 4: Experiments)

   "These MLP layers are learned during the finetuning stage. Specifically, we formulate VQA, VCR, NLVR<sup>2</sup>, Visual Entailment and RE Comprehension as classification problems and minimize the cross-entropy over the ground-truth answers/responses. For Image-Text Retrieval, we formulate it as a ranking problem." (Section 4.1: Downstream Tasks)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{6\ \text{tasks}}{6\ \text{models}} = 1
}
$$
