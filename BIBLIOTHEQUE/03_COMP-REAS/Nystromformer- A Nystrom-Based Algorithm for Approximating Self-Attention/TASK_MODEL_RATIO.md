1. **Number of distinct tasks evaluated: 11**

> "We consider the datasets of SST-2 (Socher et al. 2013), MRPC (Dolan and Brockett 2005), QNLI (Rajpurkar et al. 2016), QQP (Chen et al. 2018), and MNLI (Williams, Nangia, and Bowman 2018) in GLUE benchmark and IMDB reviews (Maas et al. 2011)." (Section: **Fine-tuning on Downstream NLP tasks**, **Datasets and metrics**)

> "We consider the LRA benchmark (Tay et al. 2020) with tasks of Listops (Nangia and Bowman 2018), byte-level IMDb reviews text classification (Maas et al. 2011), byte-level document retrieval (Radev et al. 2013), image classification on sequences of pixels (Krizhevsky, Hinton et al. 2009), and Pathfinder (Linsley et al. 2018)." (Section: **Long Range Arena (LRA) Benchmark**, **Datasets and metrics**)

2. **Number of trained model instances required to cover all tasks: 11**

> "Our second experiment is designed to test the generalization ability of our model on downstream NLP tasks. To this end, we fine-tune the pretrained model across several NLP tasks." (Section: **Fine-tuning on Downstream NLP tasks**)

> "We fine-tune our pre-trained model on GLUE benchmark datasets and IMDB reviews respectively and report its final performance." (Section: **Fine-tuning on Downstream NLP tasks**, **Implementation details**)

> "We follow the evaluation protocol from (Tay et al. 2020), including the train/test splits, and report the classification accuracy for each task, as well as the average accuracy across all tasks." (Section: **Long Range Arena (LRA) Benchmark**, **Datasets and metrics**)

A single jointly trained model instance that simultaneously covers all listed tasks: Not specified in the paper.

3. **Task–Model Ratio**

$$
\boxed{
\frac{11\ \text{tasks}}{11\ \text{models}} = 1
}
$$
