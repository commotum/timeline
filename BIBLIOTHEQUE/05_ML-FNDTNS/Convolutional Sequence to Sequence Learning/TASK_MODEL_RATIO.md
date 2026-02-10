1. **Number of distinct tasks evaluated:** 4

- "We consider three major WMT translation tasks as well as a text summarization task." (Section 4.1, Datasets)
- "WMT'16 English-Romanian." (Section 4.1, Datasets)
- "WMT'14 English-German." (Section 4.1, Datasets)
- "WMT'14 English-French." (Section 4.1, Datasets)
- "Abstractive summarization." (Section 4.1, Datasets)

2. **Number of trained model instances required to cover all tasks:** 4 models

- English-Romanian translation: "This instance of our architecture has 20 layes in the encoder and 20 layers in the decoder, both using kernels of width 3 and hidden size 512 throughout." (Section 5.1, Recurrent vs. Convolutional Models)
- English-German translation: "Our encoder has 15 layers and the decoder has 15 layers, both with 512 hidden units in the first ten layers and 768 units in the subsequent three layers, all using kernel width 3." (Section 5.1, Recurrent vs. Convolutional Models)
- English-French translation: "The ConvS2S model for this experiment uses 15 layers in the encoder and 15 layers in the decoder, both with 512 hidden units in the first five layers, 768 units in the subsequent four layers, 1024 units in the next 3 layers, all using kernel width 3; the final two layers have 2048 units and 4096 units each but the they are linear mappings with kernel width 1." (Section 5.1, Recurrent vs. Convolutional Models)
- Summarization: "We use standard likelhood training for our model and a simple model with six layers in the encoder and decoder each, hidden size 256, batch size 128, and we trained on a single GPU in one night." (Section 5.7, Summarization)

3. **Task-Model Ratio**

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
