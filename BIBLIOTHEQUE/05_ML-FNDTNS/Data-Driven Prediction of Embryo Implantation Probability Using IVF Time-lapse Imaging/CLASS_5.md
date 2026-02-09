# Data-Driven Prediction of Embryo Implantation Probability Using IVF Time-lapse Imaging (Not specified in the paper.)
Source: Data-Driven Prediction of Embryo Implantation Probability Using IVF Time-lapse Imaging.md

## Core reasons
- The main contribution is a neural modeling approach for prediction (CNN autoencoder + LSTM), rather than a positional-encoding, dimensional-lifting, or reasoning-compute mechanism proposal.
- The dataset is used to train/evaluate the model against embryologists, but the paper’s focus is model performance, not introducing a benchmark or measurement framework.

## Evidence extracts
- "In this paper we describe a novel data-driven system trained to directly predict embryo implantation probability from embryogenesis time-lapse imaging videos." (Section: Abstract)
- "A CNN autoencoder was trained with the  $L_2$  loss on the individual frames from the unlabeled videos. The encoder comprising 10 layers was used to produce a 968-dimensional embedding per frame. An LSTM network was trained on the 4,087 graded videos" (Section: 3. Methods)
- "In this paper we show that, using a small number of labeled samples, we built an embryo outcome prediction model that outperforms a panel of expert embryologists." (Section: 5. Discussion)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
