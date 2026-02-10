1. **Number of distinct tasks evaluated:** 5

   "We adapt the pre-trained model to five downstream V+L tasks." (Section 5: Downstream V+L Tasks)

   "Image-Text Retrieval contains two subtasks: image-to-text retrieval (TR) and text-to-image retrieval (IR)." (Section 5: Downstream V+L Tasks)

   "**Visual Entailment** (SNLI-VE<sup>5</sup> [51]) is a fine-grained visual reasoning task to predict whether the relationship between an image and a text is entailment, neutral, or contradictory." (Section 5: Downstream V+L Tasks)

   "**Visual Question Answering** (VQA [52]) requires the model to predict an answer given an image and a question." (Section 5: Downstream V+L Tasks)

   "Natural Language for Visual Reasoning (NLVR<sup>2</sup> [19]) requires the model to predict whether a text describes a pair of images." (Section 5: Downstream V+L Tasks)

   "**Visual Grounding** aims to localize the region in an image that corresponds to a specific textual description." (Section 5: Downstream V+L Tasks)

2. **Number of trained model instances required to cover all tasks:** 5

   "We introduce each task and our fine-tuning strategy below." (Section 5: Downstream V+L Tasks)

   "We evaluate ALBEF on the Flickr30K [49] and COCO benchmarks, and fine-tune the pre-trained model using the training samples from each dataset." (Image-Text Retrieval, Section 5: Downstream V+L Tasks)

   "We follow UNITER [2] and consider VE as a three-way classification problem, and predict the class probabilities using a multi-layer perceptron (MLP) on the multimodal encoder's representation of the [CLS] token." (Visual Entailment, Section 5: Downstream V+L Tasks)

   "Specifically, we use a 6-layer transformer decoder to generate the answer." (Visual Question Answering, Section 5: Downstream V+L Tasks)

   "We extend our multimodal encoder to enable reasoning over two images." and "For NLVR<sup>2</sup>, we perform an additional pre-training step to prepare the new multimodal encoder for encoding an image-pair." (NLVR<sup>2</sup>, Section 5: Downstream V+L Tasks)

   "We perform experiments on the RefCOCO+ [56] dataset, and fine-tune the model using only image-text supervision following the same strategy as image-text retrieval." (Visual Grounding, Section 5: Downstream V+L Tasks)

3. **Task–Model Ratio**

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$
