1. **Number of distinct tasks evaluated:** 4

Verbatim evidence:
- "We present results in three domains: language modeling (Section 4.1), speech recognition (Section 4.2), machine translation (Section 4.3), and image caption generation (Section 4.4)." (Section 4, "EXPERIMENTS")
- "## 4.1 Language modeling" (Section 4.1)
- "### 4.2 Speech recognition" (Section 4.2)
- "### 4.3 MACHINE TRANSLATION" (Section 4.3)
- "### 4.4 IMAGE CAPTION GENERATION" (Section 4.4)

2. **Number of trained model instances required to cover all tasks:** 4 models

Verbatim evidence:
- "We conducted word-level prediction experiments on the Penn Tree Bank (PTB) dataset Marcus et al. (1993), which consists of 929k training words, 73k validation words, and 82k test words." (Section 4.1, "Language modeling")
- "We trained regularized LSTMs of two sizes; these are denoted the medium LSTM and large LSTM." (Section 4.1, "Language modeling")
- "We report the performance of an LSTM on an internal Google Icelandic Speech dataset, which is relatively small (93k utterances), so overfitting is a great concern." (Section 4.2, "Speech recognition")
- "Thus, the LSTM is trained on concatenations of source sentences and their translations Sutskever et al. (2014) (see also Cho et al. (2014))." (Section 4.3, "MACHINE TRANSLATION")
- "We ran an LSTM on the WMT'14 English to French dataset, on the \"selected\" subset from Schwenk (2014) which has 340M French words and 304M English words." (Section 4.3, "MACHINE TRANSLATION")
- "We applied the dropout variant to the image caption generation model of Vinyals et al. (2014)." (Section 4.4, "IMAGE CAPTION GENERATION")
- "We test our dropout scheme on LSTM as the convolutional neural network is not trained on the image caption dataset because it is not large (MSCOCO (Lin et al., 2014))." (Section 4.4, "IMAGE CAPTION GENERATION")
- Single jointly trained model instance that performs all four tasks: "Not specified in the paper."

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
