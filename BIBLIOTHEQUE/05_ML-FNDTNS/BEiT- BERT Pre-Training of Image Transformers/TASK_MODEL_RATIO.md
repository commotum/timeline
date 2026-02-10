1. **Number of distinct tasks evaluated:** 2

> "We perform self-supervised learning and then fine-tune the pretrained BEIT on two downstream tasks, i.e., image classification, and semantic segmentation." (Section 1 Introduction)

> "We conduct full fine-tuning experiments on image classification and semantic segmentation." (Section 3 Experiments)

2. **Number of trained model instances required to cover all tasks:** 2

> "After pre-training BEIT, we append a task layer upon the Transformer, and fine-tune the parameters on downstream tasks, like BERT. We take image classification and semantic segmentation as examples in our work." (Section 2.6 Fine-Tuning BEIT on Downstream Vision Tasks)

> "Image classification. For image classification tasks, we directly employ a simple linear classifier as the task layer." (Section 2.6 Fine-Tuning BEIT on Downstream Vision Tasks)

> "**Semantic segmentation.** For semantic segmentation, we follow the task layer used in SETR-PUP [ZLZ<sup>+</sup>20]. To be specific, we use pretrained BEIT as a backbone encoder, and incorporate several deconvolution layers as decoder to produce segmentation." (Section 2.6 Fine-Tuning BEIT on Downstream Vision Tasks)

3. **Task–Model Ratio**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
