1. Number of distinct tasks evaluated: 5. "We transfer our pretrained ViLBERT model to a set of four established vision-and-language tasks (see examples in Fig.4) and one diagnostic task." (Sec. 3.2 Vision-and-Language Transfer Tasks)
2. Number of trained model instances required to cover all tasks: 5. "We follow a fine-tuning strategy where we modify the pretrained base model to perform the new task and then train the entire model end-to-end. In all cases, the modification is trivial – typically amounting to learning a classification layer." (Sec. 3.2 Vision-and-Language Transfer Tasks) "The previous tasks are all transfer tasks that include dataset specific fine-tuning. In this 'zero-shot' task, we directly apply the pretrained the multi-modal alignment prediction mechanism to caption-based image retrieval in Flickr30k [26] without fine-tuning (thus the description as 'zero-shot')." (Sec. 3.2 'Zero-shot' Caption-Based Image Retrieval) "We directly use the ViLBERT model trained on Conceptual Captions dataset described in Sec. 3.1." (Sec. 3.2 'Zero-shot' Caption-Based Image Retrieval)

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$
