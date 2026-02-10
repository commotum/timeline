1. **Number of distinct tasks evaluated:** 20
"Empirically, under comparable experiment settings, XLNet outperforms BERT on 20 tasks, often by a large margin, including question answering, natural language inference, sentiment analysis, and document ranking." (Abstract)
"SQuAD is a large-scale reading comprehension dataset with two tasks." (A.3.2 SQuAD)
"Following previous work on text classification [39, 23], we evaluate XLNet on the following benchmarks: IMDB, Yelp-2, Yelp-5, DBpedia, AG, Amazon-2, and Amazon-5." (A.3.3 Text classification Datasets)
"The GLUE dataset [34] is a collection of 9 natural language understanding tasks." (A.3.4 GLUE Dataset)
"Following the setting in previous work [8], we use the ClueWeb09-B dataset to evaluate the performance on document ranking." (A.3.5 ClueWeb09-B Dataset)

2. **Number of trained model instances required to cover all tasks:** 20
"Typically, these methods first pretrain neural networks on large-scale unlabeled text corpora, and then finetune the models or representations on downstream tasks." (1 Introduction)
"Our finetuning procedure follows BERT [10] except otherwise specified." (3.1 Pretraining and Implementation)
"Only single-task training is employed for the four large datasets." (A.3.4 GLUE Dataset)
"To finetune an XLNet on SQuAD2.0, we jointly apply a logistic regression loss for answerability prediction similar to classification tasks and a standard span extraction loss for question answering [10]." (A.3.2 SQuAD)
"We use a pretrained XLNet to extract word embeddings for the documents and queries without finetuning, and employ a kernel pooling network [36] to rank the documents." (A.3.5 ClueWeb09-B Dataset)

3. **Task-Model Ratio = (1) / (2)**

$$
\boxed{
\frac{20\ \text{tasks}}{20\ \text{models}} = 1
}
$$
