1. **Number of distinct tasks evaluated:** 1

   Verbatim evidence:
   - "In this paper we describe a novel data-driven system trained to directly predict embryo implantation probability from embryogenesis time-lapse imaging videos." (Abstract)
   - "We introduce a novel machine learning algorithm, referred to as *Ubar*, that takes timelapse images as the input and predicts embryo implantation probability." (1. Introduction)
   - "Receiver operating characteristic (ROC) curves were calculated for both UBar predictions and panel scores, with thresholds between 0 and 1 (UBar) or 1 and 5 (panel) and are depicted in Figure 1A." (4. Results)
   - "An LSTM network was trained on the 4,087 graded videos receiving the embeddings of the sequence of frames and predicting the embryologist grade distribution." (3. Methods)

   Whether embryologist-grade prediction is evaluated as a separate reported task outcome: **Not specified in the paper.**

2. **Number of trained model instances required to cover all tasks:** 1

   Verbatim evidence:
   - "The same network was used with a different binary head to predict the implantation probability on the 272 videos with known implantation data." (3. Methods)

3. **Task–Model Ratio**

$$
\boxed{
\frac{1\ \text{task}}{1\ \text{model}} = 1
}
$$
