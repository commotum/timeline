1. Number of distinct tasks evaluated: 11.
Counted tasks: 1 machine translation (WMT 2014 English-German), 1 masked language-modeling pre-training (BookCorpus + Wikipedia), 6 GLUE tasks (MRPC, SST-2, QNLI, STS-B, QQP, MNLI), 1 language modeling task on Enwik8, 1 Chinese pre-training task, 1 CAIL2019-SCM semantic text matching task.
"We choose the standard WMT 2014 English-German datasetBojar et al. [2014], which consists of approximately 4.5 million sentence pairs." (Section 4.1.1)
"We use the BookCorpus Zhu et al. [2015] and the Wikipedia Corpus Foundation [2021] from Huggingface Datasets library (Apache License 2.0) for pre-training." (Section 4.2.1)
"We use the masked language-modeling (MLM) loss values of the training process as an evaluation metric." (Section 4.2.1)
"We look at several datasets from GLUE, i.e. MRPC Dolan and Brockett [2005], SST-2 Socher et al. [2013], QNLI Rajpurkar et al. [2016], STS-B Al-Natsheh [2017], QQP Chen et al. [2018b] and MNLI Williams et al. [2018]." (Section 4.3.1)
"We carry out tests on the Enwik8 dataset Mahoney [2006], which is from English Wikipedia that includes markup, special characters and text in other languages in addition to English text." (Section 4.4.1)
"We pre-train RoFormer on approximately 34GB of data collected from Chinese Wikipedia, news and forums." (Section 4.5.2)
"We choose Chinese AI and Law 2019 Similar Case Matching (CAIL2019-SCM)Xiao et al. [2019] dataset to illustrate the ability of RoFormer in dealing with long texts, i.e., semantic text matching." (Section 4.5.3)
"The task is to predict whether the pair (A, B) is closer than (A, C) under a predefined similarity measure." (Section 4.5.3)

2. Number of trained model instances required to cover all tasks: 11.
Counted models: 1 translation model, 1 MLM pre-training model, 6 task-specific GLUE fine-tunes, 1 Enwik8 language-modeling PerFormer, 1 Chinese pre-training model, 1 CAIL2019-SCM task model.
"We train the baseline model and our RoFormer under the same settings and report the results in Table (1)." (Section 4.1.3)
"We train both BERT and RoFormer with batch size 64 and maximum sequence length of 512 for 100k steps." (Section 4.2.2)
"Consistent with the previous experiments, we fine-tune the weights of our pre-trained RoFormer across various GLUE tasks in order to evaluate its generalization ability on the downstream NLP tasks." (Section 4.3)
"We use Huggingface Transformers library (Apache License 2.0)Wolf et al. [2020] to fine-tune each of the aforementioned downstream tasks for 3 epochs, with a maximum sequence length of 512, batch size of 32 and learning rates 2,3,4,5e-5." (Section 4.3.2)
"We carry out tests on the Enwik8 dataset Mahoney [2006], which is from English Wikipedia that includes markup, special characters and text in other languages in addition to English text. We incorporate RoPE into the 12 layer char-based PerFormer with 768 dimensions and 12 heads<sup>2</sup>." (Section 4.4.1)
"We pre-train RoFormer on approximately 34GB of data collected from Chinese Wikipedia, news and forums." (Section 4.5.2)
"We apply the pre-trained RoFormer model to CAIL2019-SCM with different input lengths." (Section 4.5.4)

$$
\boxed{
\frac{11\ \text{tasks}}{11\ \text{models}} = 1
}
$$
