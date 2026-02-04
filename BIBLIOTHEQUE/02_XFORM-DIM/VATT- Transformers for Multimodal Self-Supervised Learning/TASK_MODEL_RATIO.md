Number of distinct tasks evaluated: 4
Evidence (Abstract): "We train VATT end-to-end from scratch using multimodal contrastive losses and evaluate its performance by the downstream tasks of video action recognition, audio event classification, image classification, and text-to-video retrieval."

Number of trained model instances required to cover all tasks: 4
Evidence (A.1.2 Downstream): "Since UCF101 and HMDB51 are small datasets compared to the size of our model, we freeze the vision backbone and use its outputs to train a linear classifier."
Evidence (A.1.2 Downstream): "We also use AudioSet to fine-tune our audio backbone initialized from the pre-trained checkpoint."
Evidence (A.1.2 Downstream): "We fine-tune the last checkpoint of the vision Transformer on ImageNet [22] with no modification to our architecture or the tokenization pipeline."
Evidence (A.1.2 Downstream): "We evaluate the quality of our video-text common space representations by *zero-shot* text-to-video retrieval on two of the most established datasets in this area: YouCook2 [109] and MSR-VTT [98] with 3.1k and 1k video-text pairs, respectively."

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
