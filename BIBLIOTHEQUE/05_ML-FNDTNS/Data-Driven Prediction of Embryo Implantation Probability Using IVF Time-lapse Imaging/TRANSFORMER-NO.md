# Data-Driven Prediction of Embryo Implantation Probability Using IVF Time-lapse Imaging (Year not specified)
Source: Data-Driven Prediction of Embryo Implantation Probability Using IVF Time-lapse Imaging.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract and auxiliary analyses describe a CNN autoencoder plus LSTM-based pipeline for prediction, with no Transformer block or self-attention mechanism identified as central.
- Reported model-family cues in auxiliary files explicitly characterize attention as static/inferred and focus on CNN/LSTM heads for the main implantation prediction task.

## Evidence
- "In this paper we describe a novel data-driven system trained to directly predict embryo implantation probability from embryogenesis time-lapse imaging videos." (Abstract, Data-Driven Prediction of Embryo Implantation Probability Using IVF Time-lapse Imaging.md)
- "A CNN autoencoder was trained with the  $L_2$  loss on the individual frames from the unlabeled videos." (Evidence section, TASK-DOMAINS.md; cites Section 3. Methods)
- "An LSTM network was trained on the 4,087 graded videos receiving the embeddings of the sequence of frames and predicting the embryologist grade distribution." (Evidence section, TASK-DOMAINS.md; cites Section 3. Methods)
- "The same network was used with a different binary head to predict the implantation probability on the 272 videos with known implantation data." (TASK_MODEL_RATIO.md; cites 3. Methods)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence TRANSFORMER-NO decision (CNN autoencoder + LSTM, no Transformer/self-attention core).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient; extending-dimensions analysis file was unavailable (MISSING).
