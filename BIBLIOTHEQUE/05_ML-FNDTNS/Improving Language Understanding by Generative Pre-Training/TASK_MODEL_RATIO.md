1. **Number of distinct tasks evaluated:** 12

> "Our general task-agnostic model outperforms discriminatively trained models that use architectures specifically crafted for each task, significantly improving upon the state of the art in 9 out of the 12 tasks studied." (Abstract)

> "Table 1: A list of the different tasks and datasets used in our experiments." (Section 4.1 Setup)
>
> "| Natural language inference | SNLI [5], MultiNLI [66], Question NLI [64], RTE [4], SciTail [25]       |" (Table 1, Section 4.1 Setup)
>
> "| Question Answering         | RACE [30], Story Cloze [40]                                             |" (Table 1, Section 4.1 Setup)
>
> "| Sentence similarity        | MSR Paraphrase Corpus [14], Quora Question Pairs [9], STS Benchmark [6] |" (Table 1, Section 4.1 Setup)
>
> "| Classification             | Stanford Sentiment Treebank-2 [54], CoLA [65]                           |" (Table 1, Section 4.1 Setup)

2. **Number of trained model instances required to cover all tasks:** 12

> "We demonstrate that large gains on these tasks can be realized by generative pre-training of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task." (Abstract)

> "Subsequently, we adapt these parameters to a target task using the corresponding supervised objective." (Section 1 Introduction)

> "Overall, the only extra parameters we require during fine-tuning are  $W_y$ , and embeddings for delimiter tokens (described below in Section 3.3)." (Section 3.2 Supervised fine-tuning)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{12\ \text{tasks}}{12\ \text{models}} = 1
}
$$
