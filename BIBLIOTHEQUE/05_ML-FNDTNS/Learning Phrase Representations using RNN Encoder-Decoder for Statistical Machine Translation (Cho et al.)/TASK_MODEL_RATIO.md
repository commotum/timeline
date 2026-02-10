1. **Number of distinct tasks evaluated:** 1

   "The proposed RNN Encoder—Decoder with a novel hidden unit is empirically evaluated on the task of translating from English to French." (Section 1: Introduction)

   "We evaluate our approach on the English/French translation task of the WMT'14 workshop." (Section 4: Experiments)

   "We evaluated the proposed model with the task of statistical machine translation, where we used the RNN Encoder–Decoder to score each phrase pair in the phrase table." (Section 5: Conclusion)

2. **Number of trained model instances required to cover all tasks:** 1

   "We train the model to learn the translation probability of an English phrase to a corresponding French phrase." (Section 1: Introduction)

   "Here we propose to train the RNN Encoder–Decoder (see Sec. 2.2) on a table of phrase pairs and use its scores as additional features in the log-linear model in Eq. (9) when tuning the SMT decoder." (Section 3.1: Scoring Phrase Pairs with RNN Encoder–Decoder)

   "In this paper, thus, we only consider rescoring the phrase pairs in the phrase table." (Section 3.1: Scoring Phrase Pairs with RNN Encoder–Decoder)

3. **Task–Model Ratio**

$$
\boxed{
\frac{1\ \text{tasks}}{1\ \text{models}} = 1
}
$$
