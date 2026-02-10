1. **Number of distinct tasks evaluated:** 14

"We evaluate our models on the SQuAD, Google-RE and T-REx subsets of the LAMA benchmark (Petroni et al., 2019)." (Section 4.2.1, LAMA)

"We test mathematical reasoning abilities on ASDiv (Miao et al., 2020), SVAMP (Patel et al., 2021) and the MAWPS benchmark (Koncel-Kedziorski et al., 2016)." (Section 4.2.2, Math Datasets)

"We look at Web Questions (Berant et al., 2013), Natural Questions (Kwiatkowski et al., 2019) and TriviaQA (Joshi et al., 2017), the three question answering datasets considered by Brown et al. (2020)." (Section 4.2.3, Question Answering)

"We evaluate Toolformer and all baseline models on MLQA (Lewis et al., 2019), a multilingual question-answering benchmark." (Section 4.2.4, Multilingual Question Answering)

"To investigate the calendar API's utility, we evaluate all models on TEMPLAMA (Dhingra et al., 2022) and a new dataset that we call DATESET." (Section 4.2.5, Temporal Datasets)

"we evaluate our models on two language modeling datasets: WikiText (Merity et al., 2017) and a subset of 10,000 randomly selected documents from CCNet (Wenzek et al., 2020) that were not used during training." (Section 4.3, Language Modeling)

2. **Number of trained model instances required to cover all tasks:** 1

"Throughout all of our experiments, we use a subset of CCNet (Wenzek et al., 2020) as our language modeling dataset C and GPT-J (Wang and Komatsuzaki, 2021) as our language model M." (Section 4.1, Experimental Setup)

"Toolformer: GPT-J finetuned on  C^* , our subset of CCNet augmented with API calls." (Section 4.1, Experimental Setup)

"In all cases, we consider a prompted zero-shot setup" (Section 4.2, Downstream Tasks)

"Toolformer (disabled): The same model as Toolformer, but API calls are disabled during decoding." (Section 4.1, Experimental Setup)

3. **Task–Model Ratio**

$$
\boxed{
\frac{14\ \text{tasks}}{1\ \text{model}} = 14
}
$$
