# VideoBERT: A Joint Model for Video and Language Representation Learning (Not specified in the paper.)
Source: VideoBERT- A Joint Model for Video and Language Representation Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Masked token prediction (cloze) | Linguistic and/or visual token sequences with masked positions | 1D (t); 3D (x, y, z) or (x, y, t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Imputed masked tokens (text or visual tokens) | 1D (t); 3D (x, y, z) or (x, y, t) | Capped (inferred) |
| Linguistic-visual alignment classification | Paired linguistic sentence and visual sentence | 1D (t); 3D (x, y, z) or (x, y, t) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Temporal alignment label (aligned vs. not aligned) | 0D | Fixed |
| Text-to-video prediction | Recipe/spoken-word sentence sequence | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Generated sequence of video tokens | 3D (x, y, z) or (x, y, t) | Capped (inferred) |
| Video-to-video future forecasting | Video token(s) / visual-token sequence | 3D (x, y, z) or (x, y, t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Forecasted future video token(s) | 3D (x, y, z) or (x, y, t) | Capped (inferred) |
| Zero-shot action classification | Sequence of visual tokens (video clip) with a fixed masked text prompt | 3D (x, y, z) or (x, y, t); 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Verb and noun labels from predicted masked slots | 0D | Fixed |
| Video captioning | Video segments represented by VideoBERT (and S3D) features | 3D (x, y, z) or (x, y, t) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Generated caption token sequence | 1D (t) | Capped (inferred) |

## Summary
The OCR supports six distinct task intents spanning self-supervised pretraining objectives (cloze mask completion and linguistic-visual alignment), generative/predictive uses (text-to-video prediction and video forecasting), and downstream applications (zero-shot action classification and video captioning). Inputs and outputs span language token streams (1D (t)), spatiotemporal video domains represented via visual tokens (3D (x, y, z) or (x, y, t)), and label outputs (0D). Dynamics are mostly treated as capped from segmentation/template-based finite sequences, with fixed outputs for binary/slot-based classification. Attention and state labels are inferred from the described transformer-based processing and explicit use of [CLS]-derived representations.

## Evidence
### Task: Masked token prediction (cloze)
- "Figure 3: Illustration of VideoBERT in the context of a video and text masked token prediction, or *cloze*, task." (Figure 3 / Section 3.2)
- "For text-only and video-only, the standard mask-completion objectives are used for training the model." (Section 3.2)
- "In the first application, we treat it as a probabilistic model, and ask it to predict or impute the symbols that have been MASKed out." (Section 3.2)
- Inference: `Capped`, `Static`, and `Direct` are inferred because the model operates on segmented finite token sequences ("we treat video tokens that fall into that time period as a segment... we simply treat 16 tokens as a segment," Section 4.2) with a bidirectional transformer over provided inputs (Section 3.1), and this task is framed as direct masked-symbol imputation (Section 3.2).

### Task: Linguistic-visual alignment classification
- "We propose a linguistic-visual alignment task, where we use the final hidden state of the <code>[CLS]</code> token to predict whether the linguistic sentence is temporally aligned with the visual sentence." (Section 3.2)
- "For text-video, we use the linguistic-visual alignment classification objective described above." (Section 3.2)
- Inference: `Capped` is inferred from sentence/video-segment preprocessing (Section 4.2); `Static` is inferred from transformer processing over a provided sequence (Section 3.1); `Constructed` is inferred from explicit use of a learned aggregate `[CLS]` representation to drive prediction (Sections 3.1 and 3.2).

### Task: Text-to-video prediction
- "For example, we can perform text-to-video prediction, which can be used to automatically illustrate a set of instructions (such as a recipe), as shown in the top examples of Figure 1 and 2." (Section 1)
- "Figure A3: Visualizations for text to video prediction. In particular, we make small changes to the input text, and compare how the generated video tokens vary. We show top 2 retrieved video tokens for each text query." (Figure A3)
- Inference: Output dimension is marked `3D (x, y, z) or (x, y, t)` because visual tokens are derived from spatio-temporal video features (Sections 1 and 4.2). `Capped`, `Static`, and `Direct` are inferred from finite sequence generation (`x_{1:T}`) with transformer-based conditional prediction over provided input tokens.

### Task: Video-to-video future forecasting
- "We can also use our model in a \"unimodal\" fashion. For example, the implied marginal distribution p(x) is a language model for visual words, which we can use for longrange forecasting." (Section 1)
- "Given a video token, we show the top three future tokens forecasted by VideoBERT at different time scales." (Figure 1)
- "Figure A2: Visualizations for video to video prediction. Given an input video token, we show the top 3 predicted video tokens 2 steps away in the future." (Figure A2)
- Inference: `Capped`, `Static`, and `Direct` are inferred from finite-horizon token forecasting examples and transformer-based token modeling; the 3D label is inferred because these tokens represent quantized video features (Sections 1, 3.2, 4.2).

### Task: Zero-shot action classification
- "Once pretrained, the VideoBERT model can be used for \"zero-shot\" classification on novel datasets, such as YouCook II" (Section 4.4)
- "More precisely, we want to compute p(y|x) where x is the sequence visual tokens, and y is a sequence of words." (Section 4.4)
- "Since the model is trained to predict sentences, we define y to be the fixed sentence, \"now let me show you how to <code>[MASK]</code> the <code>[MASK]</code>,\" and extract the verb and noun labels from the tokens predicted in the first and second masked slots, respectively." (Section 4.4)
- Inference: `Capped` and `Static` are inferred from segmented tokenized inputs processed by BERT-style transformer context (Sections 3.1 and 4.2). `Direct` is inferred because this application is defined as direct masked-token prediction for label extraction (Section 3.2 and Section 4.4). Output `Fixed`/`0D` is inferred from the two fixed masked label slots (verb, noun).

### Task: Video captioning
- "We can also perform the more traditional video-to-text task of dense video captioning [10] as shown in Figure 6." (Section 1)
- "We evaluate the extracted features on video captioning, following the setup from [39], where the ground truth video segmentations are used to train a supervised model mapping video segments to captions." (Section 4.6)
- "We use the same model that they do, namely a transformer encoder-decoder, but we replace the inputs to the encoder with the features derived from VideoBERT described above." (Section 4.6)
- Inference: `Capped` and `Static` are inferred from segmented video inputs and transformer encoder-decoder processing of provided sequences (Sections 4.2 and 4.6). `Constructed` is inferred because the task explicitly uses learned internal representations ("extract the predicted representation ... for the [CLS] token") as downstream state/features (Section 3.2 and Section 4.6).
