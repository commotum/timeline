# Data-Driven Prediction of Embryo Implantation Probability Using IVF Time-lapse Imaging (Year not specified in the paper)
Source: Data-Driven Prediction of Embryo Implantation Probability Using IVF Time-lapse Imaging.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frame reconstruction for representation learning (inferred) | Individual frames from unlabeled embryo time-lapse videos | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Reconstructed embryo video frames (inferred) | 2D (x, y) (inferred) | Fixed (inferred) |
| Embryologist grade distribution prediction | Embedding sequences of graded embryo videos | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Embryologist grade distribution | 0D (inferred) | Fixed (inferred) |
| Embryo implantation probability prediction | Time-lapse images/videos of embryo development | 3D (x, y, t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Embryo implantation probability | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers three model-handled tasks: frame-level autoencoding, video-sequence prediction of embryologist grade distribution, and implantation probability prediction from embryo time-lapse imaging. Input modalities span 2D frames and spatiotemporal video/temporal sequences (2D and 3D/1D task views), while outputs are either reconstructed frames or point-like predictive targets. Based on the described CNN encoder and LSTM heads, attention is static and state is constructed for all rows (inferred). Dynamics are fixed for frame reconstruction and fixed-output prediction heads, with capped temporal input sequences inferred from finite embryo videos.

## Evidence
### Task: Frame reconstruction for representation learning (inferred)
- "A CNN autoencoder was trained with the  $L_2$  loss on the individual frames from the unlabeled videos." (Section 3. Methods)
- "The encoder comprising 10 layers was used to produce a 968-dimensional embedding per frame." (Section 3. Methods)
- Inference: The text states an autoencoder trained with frame-level $L_2$ loss, which supports a frame reconstruction task; because the model is a CNN autoencoder operating on frames, 2D fixed-size image processing, static attention, and constructed latent state are inferred from the architectural description.

### Task: Embryologist grade distribution prediction
- "An LSTM network was trained on the 4,087 graded videos receiving the embeddings of the sequence of frames and predicting the embryologist grade distribution." (Section 3. Methods)
- "...an external panel of five embryologists from various countries (India, Latvia, Ukraine, and the United States) assigned each embryo video a grade between 1 and 5..." (Section 3. Methods)
- Inference: "sequence of frames" supports temporal indexing (1D (t)); grades "between 1 and 5" support a fixed-size output target space; static attention and constructed state are inferred from the described LSTM-on-embeddings pipeline with no runtime retrieval/selection mechanism described.

### Task: Embryo implantation probability prediction
- "We introduce a novel machine learning algorithm, referred to as *Ubar*, that takes timelapse images as the input and predicts embryo implantation probability." (Section 1. Introduction)
- "The same network was used with a different binary head to predict the implantation probability on the 272 videos with known implantation data." (Section 3. Methods)
- Inference: Time-lapse videos imply a spatiotemporal input domain (3D (x, y, t)); the binary prediction head and probability target imply point-like fixed output; static attention and constructed state are inferred from the described encoder/LSTM + head architecture.
