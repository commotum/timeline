1. **Number of distinct tasks evaluated:** 2.

   "We demonstrate the advantages of GPipe by training large-scale neural networks on two different tasks with distinct network architectures: (i) Image Classification: We train a 557-million-parameter AmoebaNet model and attain a top-1 accuracy of 84.4% on ImageNet-2012, (ii) Multilingual Neural Machine Translation: We train a single 6-billion-parameter, 128-layer Transformer model on a corpus spanning over 100 languages and achieve better quality than all bilingual models." (Abstract)

2. **Number of trained model instances required to cover all tasks:** 2 models.

   "We trained this 557-million-parameter AmoebaNet-B(18, 512) on the ImageNet 2012 dataset, using the same hyper-parameters as described in [12]. The network was divided into 4 partitions. This single model achieves 84.4% top-1 and 97% top-5 validation accuracy with single-crop." (Section 4: Image Classification)

   "Our comparison is based on the performance of a single Transformer [15] trained on all language pairs in this corpus." (Section 5: Massive Massively Multilingual Machine Translation)

3. **Task–Model Ratio**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
