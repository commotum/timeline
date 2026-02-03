Number of distinct tasks evaluated: 5.

Quote (Section 3.2 Experimental Results):
> We test the performances of the pre-trained models by fine-tuning on a variety of natural language understanding (NLU) tasks, which are listed as follows. The hyperparameters of finetuning each task are shown in Table 3.
> - **CMRC** (Chinese Machine Reading Comprehension 2018) [16]: A machine reading comprehension task that returns an answer span in a given passage for a given question.
> - XNLI (Cross-lingual Natural Language Inference) [17]: The Chinese portion of XNLI, which is a version of MultiNLI where the dev and test sets have been translated (by humans) into 15 languages. XNLI is a natural language inference task. The goal of this task is to predict if the second sentence is a contradiction, entailment or neutral to the first sentence.
> - LCQMC (Large-scale Chinese Question Matching Corpus) [18]: A sentence pair matching task. Given a pair of sentences, the task is to determine if the two sentences are semantically equivalent or not.
> - **PD-NER** (People's Daily Named Entity Recognition) <sup>9</sup>: A sequence labeling task that identifies the named entities from text. The corpus is from *People's Daily*, a Chinese News Media.
> - **ChnSenti** (Chinese Sentiment Classification) <sup>10</sup>: A binary classification task which predicts if the sentiment of a given sentence is positive or negative.

Number of trained model instances required to cover all tasks: 5.

Quote (Section 3.2 Experimental Results):
> We test the performances of the pre-trained models by fine-tuning on a variety of natural language understanding (NLU) tasks, which are listed as follows. The hyperparameters of finetuning each task are shown in Table 3.

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$
