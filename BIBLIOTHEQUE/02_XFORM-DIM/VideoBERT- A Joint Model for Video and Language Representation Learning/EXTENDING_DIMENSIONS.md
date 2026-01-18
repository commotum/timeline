## 1. Basic Metadata

- Title: "VideoBERT: A Joint Model for Video and Language Representation Learning" (Title)
- Authors: "Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid" (Title page)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

"we propose a joint visual-linguistic model to learn high-level features without any explicit supervision" by applying BERT to sequences of visual and linguistic tokens. (Abstract)

---

## 3. Tasks Evaluated

### Task 1: Zero-shot action classification (verb and noun prediction)

- Task name: Zero-shot action classification
- Task type: Classification
- Dataset(s) used: YouCook II
- Domain: video (cooking instructional videos)
- Quotes:
  - "Once pretrained, the VideoBERT model can be used for \"zero-shot\" classification on novel datasets, such as YouCook II" (Section 4.4)
  - "we define y to be the fixed sentence, \"now let me show you how to [MASK] the [MASK],\" and extract the verb and noun labels from the tokens predicted in the first and second masked slots, respectively" (Section 4.4)
  - "We report the performance on the validation set of YouCook II." (Section 4.4)

### Task 2: Video captioning

- Task name: Video captioning
- Task type: Generation
- Dataset(s) used: YouCook II
- Domain: video (cooking instructional videos)
- Quotes:
  - "We evaluate the extracted features on video captioning, following the setup from [39], where the ground truth video segmentations are used to train a supervised model mapping video segments to captions." (Section 4.6)
  - "Table 3: Video captioning performance on YouCook II." (Section 4.6)

---

## 4. Domain and Modality Scope

- Evaluation domain scope: Single domain (cooking instructional videos). "we focus on cooking videos specifically" (Section 4.1) and "We evaluate VideoBERT on the YouCook II dataset" (Section 4.1)
- Modality scope: Multiple modalities (visual + linguistic). "x is a sequence of \"visual words\", and y is a sequence of spoken words" (Section 3.2)
- Domain generalization or cross-domain transfer: Not claimed. "we plan to assess our approach on other video understanding tasks, and on other domains besides cooking." (Section 5)

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Zero-shot action classification | Yes (pretrained VideoBERT) | No (zero-shot) | No separate head stated; uses masked-token prediction | "Once pretrained, the VideoBERT model can be used for \"zero-shot\" classification" and "we define y to be the fixed sentence, \"now let me show you how to [MASK] the [MASK],\"" (Section 4.4) |
| Video captioning | Yes (VideoBERT as feature extractor) | Not stated (feature extractor) | Yes, supervised transformer encoder-decoder | "We further demonstrate the effectiveness of VideoBERT when used as a feature extractor" and "We use the same model that they do, namely a transformer encoder-decoder, but we replace the inputs to the encoder with the features derived from VideoBERT" (Section 4.6) |

---

## 6. Input and Representation Constraints

- Fixed sampling and clip structure: "For each input video, we sample frames at 20 fps, and create clips from 30-frame (1.5 seconds) non-overlapping windows over the video." (Section 4.2)
- Fixed feature dimensionality: "apply 3D average pooling to obtain a 1024-dimension feature vector." (Section 4.2)
- Discrete visual vocabulary size: "We set d=4 and k=12, which yields 12^4=20736 clusters in total." (Section 4.2)
- Discrete text vocabulary size: "We use the same vocabulary provided by the authors of BERT, which contains 30,000 tokens." (Section 4.2)
- Segmentation heuristic: "when an ASR sentence is available, it is associated with starting and ending timestamps, and we treat video tokens that fall into that time period as a segment. When ASR is not available, we simply treat 16 tokens as a segment." (Section 4.2)
- Discrete token representation: "transform the raw visual data into a discrete sequence of tokens" and "generate a sequence of \"visual words\" by applying hierarchical vector quantization" (Section 3.2)
- Temporal subsampling: "we randomly pick a subsampling rate of 1 to 5 steps for the video tokens." (Section 3.2)

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Fixed or variable length: Variable/segmented, with heuristic segmentation and subsampling. "we randomly concatenate neighboring sentences into a single long sentence" and "we randomly pick a subsampling rate of 1 to 5 steps for the video tokens" (Section 3.2); "we simply treat 16 tokens as a segment" when ASR is unavailable (Section 4.2)
- Attention type: Not explicitly stated; uses a "multi-layer bidirectional transformer model" (Section 3.1)
- Computational cost management: tokenization and subsampling. "transform the raw visual data into a discrete sequence of tokens" (Section 3.2) and "we randomly pick a subsampling rate of 1 to 5 steps for the video tokens." (Section 3.2)

---

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: Absolute positional embeddings (learned position tags). "we can \"tag\" each word with its position in the sentence. The BERT model learns an embedding for each of the word tokens, as well as for these tags, and then sums the embedding vectors to get a continuous representation for each token." (Section 3.1)
- Where it is applied: Input only (summed token + position embeddings). "learns an embedding for each of the word tokens, as well as for these tags, and then sums the embedding vectors" (Section 3.1)
- Fixed vs modified: Not stated.

---

## 9. Positional Encoding as a Variable

- Core research variable vs fixed assumption: Not stated; positional encoding is described as tagging positions in BERT. "we can \"tag\" each word with its position in the sentence" (Section 3.1)
- Multiple positional encodings compared: Not stated.
- Claims PE choice is not critical or secondary: Not stated.

---

## 10. Evidence of Constraint Masking

- Model size: "Specifically, we use the BERT<sub>LARGE</sub> model released by the authors of [6], using the same backbone architecture: it has 24 layers of Transformer blocks, where each block has 1024 hidden units and 16 self-attention heads." (Section 4.3)
- Dataset size: "312K videos" and "The total duration of this dataset is 23,186 hours" (Section 4.1)
- Scaling data improves performance: "the accuracy grows monotonically as the amount of data increases, showing no signs of saturation." (Section 4.5)
- Scale/cross-modal reliance claim: "confirm that large amounts of training data and cross-modal information are critical to performance." (Abstract)

---

## 11. Architectural Workarounds

- Discretization via VQ: "transform the raw visual data into a discrete sequence of tokens" and "generate a sequence of \"visual words\" by applying hierarchical vector quantization" (Section 3.2)
- Temporal subsampling to manage variability and horizon: "we randomly pick a subsampling rate of 1 to 5 steps for the video tokens. This not only helps the model be more robust to variations in video speeds, but also allows the model to capture temporal dynamics over greater time horizons" (Section 3.2)
- Heuristic segmentation: "we treat video tokens that fall into that time period as a segment. When ASR is not available, we simply treat 16 tokens as a segment." (Section 4.2)
- Fixed clip extraction: "create clips from 30-frame (1.5 seconds) non-overlapping windows" (Section 4.2)
- Task-specific heads/decoders: "We use the same model that they do, namely a transformer encoder-decoder, but we replace the inputs to the encoder with the features derived from VideoBERT" (Section 4.6)
- Alignment classifier via [CLS]: "we use the final hidden state of the [CLS] token to predict whether the linguistic sentence is temporally aligned with the visual sentence." (Section 3.2)

---

## 12. Explicit Limitations and Non-Claims

- "This work is a first step in the direction of learning such joint representations." (Section 5)
- "For many applications, including cooking, it is important to use spatially fine-grained visual representations, instead of just working at the frame or clip level" (Section 5)
- "We also want to explicitly model visual patterns at multiple temporal scales, instead of our current approach, that skips frames but builds a single vocabulary." (Section 5)
- "Beyond improving the model, we plan to assess our approach on other video understanding tasks, and on other domains besides cooking." (Section 5)
- "We leave investigation into other ways of combining video and text to future work." (Section 3.2)
- "our model is not a generative model of pixels, but it is a generative model of features derived from pixels" (Section 2)

---

### 13. Constraint Profile (Synthesis)

**Constraint Profile:**
- Domain scope: Evaluation is confined to cooking instructional videos (YouCook II).
- Task structure: Two downstream tasks (zero-shot action classification and video captioning) on the same dataset.
- Representation rigidity: Fixed tokenization pipeline (20 fps, 30-frame clips, 1024-d features, 20,736 visual tokens, 30,000 wordpieces).
- Model sharing vs specialization: Shared pretrained VideoBERT features; captioning adds a separate transformer decoder.
- Role of positional encoding: Position tags are used; no comparison or ablation stated.

---

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates at least two tasks ("zero-shot" action classification and "video captioning") on the YouCook II cooking-video dataset (Sections 4.4 and 4.6). It does not evaluate across different domains, and it explicitly focuses on "cooking videos specifically" (Section 4.1).
