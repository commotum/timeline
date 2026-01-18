## 1. Basic Metadata

- Title: "ViViT: A Video Vision Transformer" (Title)
- Authors: "Anurag Arnab* Mostafa Dehghani* Georg Heigold Chen Sun Mario Lučić† Cordelia Schmid†" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper introduces "pure-transformer based models for video classification" that "extracts spatiotemporal tokens from the input video" and "factorise the spatial- and temporal-dimensions of the input" to handle long token sequences (Abstract).

---

## 3. Tasks Evaluated

- Task name: Video classification (Kinetics 400/600)
  - Task type: Classification
  - Dataset(s) used: Kinetics 400, Kinetics 600
  - Domain: video (YouTube)
  - Quotes: "Kinetics [34] consists of 10-second videos sampled at 25fps from YouTube. We evaluate on both Kinetics 400 and 600, containing 400 and 600 classes respectively." (4.1. Experimental Setup, Datasets)

- Task name: Action recognition (Epic Kitchens-100; verb/noun/action)
  - Task type: Classification
  - Dataset(s) used: Epic Kitchens-100
  - Domain: egocentric video (kitchen)
  - Quotes: "Epic Kitchens-100 consists of egocentric videos capturing daily kitchen activities spanning 100 hours and 90 000 clips [13]. We report results following the standard \"action recognition\" protocol. Here, each video is labelled with a \"verb\" and a \"noun\" and we therefore predict both categories using a single network with two \"heads\". The top-scoring verb and action pair predicted by the network form an \"action\", and action accuracy is the primary metric." (4.1. Experimental Setup, Datasets)

- Task name: Video classification (Moments in Time)
  - Task type: Classification
  - Dataset(s) used: Moments in Time
  - Domain: video (YouTube)
  - Quotes: "Moments in Time [44] consists of 800 000, 3-second YouTube clips that capture the gist of a dynamic scene involving animals, objects, people, or natural phenomena." (4.1. Experimental Setup, Datasets)

- Task name: Action recognition / fine-grained motion classification (Something-Something v2)
  - Task type: Classification
  - Dataset(s) used: Something-Something v2 (SSv2)
  - Domain: video (short clips)
  - Quotes: "Something-Something v2 (SSv2) [25] contains 220 000 videos, with durations ranging from 2 to 6 seconds." and "this dataset thus places more emphasis on a model's ability to recognise fine-grained motion cues." (4.1. Experimental Setup, Datasets)

---

## 4. Domain and Modality Scope

- Evaluation domain/modality: Multiple domains within the same modality (video). Evidence: "We evaluate the performance of our proposed models on a diverse set of video classification datasets" (4.1. Experimental Setup) and "The input to our network is a video clip of 32 frames" (4.1. Experimental Setup, Inference).
- Domain generalization or cross-domain transfer: Not claimed. The paper only states initialization from image models: "we initialise our video models from pretrained image models" (3.4. Initialisation by leveraging pretrained models).

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Video classification (Kinetics 400/600) | Not specified (no joint training across datasets stated) | Not specified; model initialization from image pretraining is stated | Single classification head implied | "We evaluate the performance of our proposed models on a diverse set of video classification datasets" (4.1. Experimental Setup); "We initialise our models from a ViT image model trained either on ImageNet-21K [15] (unless otherwise specified) or the larger JFT [57] dataset." (4.1. Experimental Setup); "Finally, a linear classifier is used to classify the encoded input" (3.1. Overview of Vision Transformers (ViT)). |
| Action recognition (Epic Kitchens-100; verb/noun/action) | Yes (single network with two heads) | Not specified; model initialization from image pretraining is stated | Yes (verb/noun heads) | "we therefore predict both categories using a single network with two \"heads\"" (4.1. Experimental Setup, Datasets); "We initialise our models from a ViT image model trained either on ImageNet-21K [15] (unless otherwise specified) or the larger JFT [57] dataset." (4.1. Experimental Setup). |
| Video classification (Moments in Time) | Not specified (no joint training across datasets stated) | Not specified; model initialization from image pretraining is stated | Single classification head implied | "We evaluate the performance of our proposed models on a diverse set of video classification datasets" (4.1. Experimental Setup); "We initialise our models from a ViT image model trained either on ImageNet-21K [15] (unless otherwise specified) or the larger JFT [57] dataset." (4.1. Experimental Setup); "Finally, a linear classifier is used to classify the encoded input" (3.1. Overview of Vision Transformers (ViT)). |
| Action recognition / fine-grained motion classification (Something-Something v2) | Not specified (no joint training across datasets stated) | Not specified; model initialization from image pretraining is stated | Single classification head implied | "We evaluate the performance of our proposed models on a diverse set of video classification datasets" (4.1. Experimental Setup); "We initialise our models from a ViT image model trained either on ImageNet-21K [15] (unless otherwise specified) or the larger JFT [57] dataset." (4.1. Experimental Setup); "Finally, a linear classifier is used to classify the encoded input" (3.1. Overview of Vision Transformers (ViT)). |

---

## 6. Input and Representation Constraints

- Video input representation: "We consider two simple methods for mapping a video  V in R^{T x H x W x C}  to a sequence of tokens" (3.2. Embedding video clips).
- Non-overlapping patches/tubelets: "ViT extracts N non-overlapping image patches" (3.1. Overview of Vision Transformers (ViT)) and "extract non-overlapping, spatio-temporal \"tubes\" from the input volume" (3.2. Embedding video clips).
- Token count tied to sampling and patching: "a total of  n_t · n_h · n_w  tokens will be forwarded through the transformer encoder" (3.2. Embedding video clips).
- Tubelet size constraint per experiment: "ViViT-B/16x2 denotes a ViT-Base backbone with a tubelet size of  h×w×t=16×16×2 . In all experiments, the tubelet height and width are equal." (4.1. Experimental Setup).
- Default clip length and stride: "The input to our network is a video clip of 32 frames using a stride of 2, unless otherwise mentioned" (4.1. Experimental Setup, Inference).
- Variable number of frames/tokens: "We now increase the number of frames input to the model, thereby increasing the number of tokens proportionally." (4.2. Ablation study).
- Variable spatial resolution: "We then vary the number of tokens fed into the model by increasing the spatial crop-size from the default of 224 to 320 in Tab. 4." (4.2. Ablation study).
- Fixed token dimensionality: "the token-dimensionality, d, remains fixed throughout all layers." (3.1. Overview of Vision Transformers (ViT)).
- Padding/resizing requirements: Not specified.

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; sequence length is defined as "a total of  n_t · n_h · n_w  tokens" (3.2. Embedding video clips).
- Sequence length fixed or variable: Variable; "We now increase the number of frames input to the model, thereby increasing the number of tokens proportionally." and "We then vary the number of tokens fed into the model by increasing the spatial crop-size from the default of 224 to 320" (4.2. Ablation study).
- Attention type:
  - Global (Model 1): "each transformer layer models all pairwise interactions between all spatio-temporal tokens" (3.3. Transformer Models for Video).
  - Factorised / hierarchical (Models 2-4): "two separate transformer encoders. The first, spatial encoder, only models interactions between tokens extracted from the same temporal index" and "then forwarded through a temporal encoder" (3.3. Transformer Models for Video); "we factorise the operation to first only compute self-attention spatially ... and then temporally" (3.3. Transformer Models for Video).
- Mechanisms to manage computational cost: "we propose several, efficient variants of our model which factorise the spatial- and temporal-dimensions of the input" (Abstract) and "MSA ... has quadratic complexity with respect to the number of tokens. This complexity is pertinent for video... and motivates the development of more efficient architectures" (3.3. Transformer Models for Video).

---

## 8. Positional Encoding (Critical Section)

- Mechanism: learned absolute positional embedding; "a learned positional embedding,  p in R^{N x d} , is added to the tokens to retain positional information" (3.1. Overview of Vision Transformers (ViT)).
- Where applied: input tokens; "A positional embedding  p  is added to each input token (Eq. 1)." (3.4. Initialisation by leveraging pretrained models).
- Fixed vs modified across experiments: Positional embeddings are initialized from image models and fine-tuned; "we initialise the positional embeddings by \"repeating\" them temporally" and "all tokens with the same spatial index have the same embedding which is then fine-tuned." (3.4. Initialisation by leveraging pretrained models). No task-specific modifications are stated.
- Ablated or compared against alternatives: Not stated.

---

## 9. Positional Encoding as a Variable

- Treated as a core research variable? No; it is presented as a fixed architectural component: "a learned positional embedding ... is added to the tokens" (3.1. Overview of Vision Transformers (ViT)).
- Multiple positional encodings compared? Not specified.
- Claim that PE choice is "not critical" or secondary? Not stated.

---

## 10. Evidence of Constraint Masking

- Model scale: "We consider ViT-Base (ViT-B, L=12,  N_H =12, d=768), ViT-Large (ViT-L, L=24,  N_H =16, d=1024), and ViT-Huge (ViT-H, L=32,  N_H =16, d=1280)" (4.1. Experimental Setup).
- Dataset scale: "Kinetics [34] consists of 10-second videos sampled at 25fps from YouTube... approximately 267 000 and 446 000" (4.1. Experimental Setup, Datasets); "Epic Kitchens-100 consists of egocentric videos ... 100 hours and 90 000 clips" (4.1. Experimental Setup, Datasets); "Moments in Time [44] consists of 800 000, 3-second YouTube clips" (4.1. Experimental Setup, Datasets); "Something-Something v2 (SSv2) [25] contains 220 000 videos" (4.1. Experimental Setup, Datasets).
- Scaling data/pretraining: "We initialise our models from a ViT image model trained either on ImageNet-21K [15] (unless otherwise specified) or the larger JFT [57] dataset." (4.1. Experimental Setup) and "by initialising our backbones from models pretrained on the larger JFT dataset [57], we obtain further improvements." (4.3. Comparison to state-of-the-art).
- Scaling tokens/inputs: "using smaller input tubelet sizes (and therefore more tokens) leads to consistent accuracy improvements across all of our model architectures" (4.2. Ablation study) and "increasing the spatial crop-size from the default of 224 to 320... there is a consistent increase in both accuracy and computation" (4.2. Ablation study).
- Architectural hierarchy/efficiency: "we propose several, efficient variants of our model which factorise the spatial- and temporal-dimensions of the input" (Abstract).
- Training tricks/regularisation: "we can effectively regularise the model during training and leverage pretrained image models to be able to train on comparatively small datasets" (Abstract) and "we employed several regularisation strategies" (4.2. Ablation study).

---

## 11. Architectural Workarounds

- Factorised encoder for efficiency: "This model consists of two separate transformer encoders. The first, spatial encoder, only models interactions between tokens extracted from the same temporal index" and then "forwarded through a temporal encoder" (3.3. Transformer Models for Video).
- Factorised self-attention to reduce cost: "we factorise the operation to first only compute self-attention spatially ... and then temporally" (3.3. Transformer Models for Video).
- Factorised dot-product attention: "we develop a model ... [and] factorise the multi-head dot-product attention operation instead" (3.3. Transformer Models for Video).
- Tokenization strategies to manage spatiotemporal input: "We simply sample  n_t  frames, and embed each 2D frame independently" and "extract non-overlapping, spatio-temporal \"tubes\" from the input volume" (3.2. Embedding video clips).
- Task-specific heads for multi-label action recognition: "we therefore predict both categories using a single network with two \"heads\"" (4.1. Experimental Setup, Datasets).

---

## 12. Explicit Limitations and Non-Claims

- Dependence on large data/regularization: "transformer-based models are known to only be effective when large training datasets are available" (Abstract).
- Motion understanding limitation: "Our results suggest that capturing these fine-grained motions is an area of improvement and future work for our model." (4.3. Comparison to state-of-the-art, Something-Something v2 (SSv2)).
- Future work / non-claim: "Future work is to remove our dependence on image-pretrained models, and to extend our model to more complex video understanding tasks." (5. Conclusion and Future Work).
- Explicit statements about open-world or unrestrained multi-task learning: Not stated.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Multiple datasets but single modality (video) evaluation.
> - Task structure: Video classification/action recognition across benchmarks; no joint multi-task training described.
> - Representation rigidity: Non-overlapping patches/tubelets, fixed tubelet size per experiment, fixed token dimensionality, default 32-frame clips.
> - Model sharing vs specialization: Separate dataset evaluations; Epic Kitchens uses a shared backbone with two heads for verb/noun.
> - Role of positional encoding: Learned absolute positional embeddings added at input and fine-tuned; no alternatives compared.

---

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates on "multiple video classification benchmarks including Kinetics 400 and 600, Epic Kitchens, Something-Something v2 and Moments in Time" (Abstract), which are all video datasets, and the input is a "video clip of 32 frames" (4.1. Experimental Setup, Inference). It does not claim joint multi-task training or cross-domain generalization, only evaluation across multiple video tasks.
