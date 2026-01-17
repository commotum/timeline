## 1. Basic Metadata

- Title: "AST: Audio Spectrogram Transformer" (Title)
- Authors: "Yuan Gong, Yu-An Chung, James Glass" (Title page)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

"In this paper, we answer the question by introducing the Audio Spectrogram Transformer (AST), the first convolution-free, purely attention-based model for audio classification." (Abstract)

---

## 3. Tasks Evaluated

- Task name: Audio event classification (AudioSet; weakly-labeled)
  - Task type: Classification
  - Dataset(s) used: AudioSet
  - Domain: Audio (YouTube audio clips; audio spectrograms)
  - Quotes: "In this section, we focus on evaluating the AST on AudioSet (Section 3.1) as weakly-labeled audio event classification is one of the most challenging audio classification tasks." (Section 3) "AudioSet [15] is a collection of over 2 million 10-second audio clips excised from YouTube videos and labeled with the sounds that the clip contains from a set of 527 labels." (Section 3.1.1)

- Task name: Environmental sound classification (ESC-50)
  - Task type: Classification
  - Dataset(s) used: ESC-50
  - Domain: Environmental audio
  - Quotes: "The ESC-50 [16] dataset consists of 2,000 5-second environmental audio recordings organized into 50 classes." (Section 3.2)

- Task name: Speech command classification (Speech Commands V2; 35-class)
  - Task type: Classification
  - Dataset(s) used: Speech Commands V2
  - Domain: Speech audio
  - Quotes: "Speech Commands V2 [17] is a dataset consists of 105,829 1-second recordings of 35 common speech commands." (Section 3.2) "We focus on the 35-class classification task," (Section 3.2)

---

## 4. Domain and Modality Scope

- Evaluation domain scope: Multiple domains within the same modality (audio). Evidence: "we evaluate AST on a variety of audio classification tasks and datasets including AudioSet [15], ESC-50 [16] and Speech Commands [17]." (Section 1. Introduction) "content varies from speech (Speech Commands) to non-speech (AudioSet and ESC-50)" (Section 3.2)
- Multiple modalities? No (evaluation is audio only). Evidence: "audio classification" (Abstract) and "audio spectrograms" (Abstract).
- Domain generalization or cross-domain transfer claimed? Yes, cross-modality transfer learning is claimed. Evidence: "which motivates us to apply cross-modality transfer learning to AST since images and audio spectrograms have similar formats." (Section 2.2)

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| AudioSet audio event classification | Not specified (architecture shared across tasks). | Not explicitly stated; ImageNet pretraining used. | Not specified. | "models we use for all aforementioned tasks have the same architecture while the input lengths vary from 1 sec. (Speech Commands) to 10 sec. (AudioSet)." (Section 1. Introduction); "we use ImageNet pretraining (as described in Section 2.2)" (Section 3.1.1). |
| ESC-50 environmental sound classification | Not specified (architecture shared across tasks). | Not explicitly stated; ImageNet and/or AudioSet pretraining used. | Not specified. | "models we use for all aforementioned tasks have the same architecture while the input lengths vary from 1 sec. (Speech Commands) to 10 sec. (AudioSet)." (Section 1. Introduction); "we train an AST model with only ImageNet pretraining (AST-S) and an AST model with ImageNet and AudioSet pretraining (AST-P)." (Section 3.2) |
| Speech Commands V2 speech command classification | Not specified (architecture shared across tasks). | Not explicitly stated; ImageNet and/or AudioSet pretraining used. | Not specified. | "models we use for all aforementioned tasks have the same architecture while the input lengths vary from 1 sec. (Speech Commands) to 10 sec. (AudioSet)." (Section 1. Introduction); "we train an AST model with only ImageNet pretraining (AST-S) and an AST model with ImageNet and AudioSet pretraining (AST-P)." (Section 3.2) |

---

## 6. Input and Representation Constraints

- Input representation and dimensionality: "the input audio waveform of tseconds is converted into a sequence of 128-dimensional log Mel filterbank (fbank) features computed with a 25ms Hamming window every 10ms." (Section 2.1) "This results in a  $128 \times 100t$  spectrogram as input to the AST." (Section 2.1)
- Fixed patch size and overlap: "We then split the spectrogram into a sequence of N 16  $\times$  16 patches with an overlap of 6 in both time and frequency dimension, where  $N = 12 \lceil (100t - 16)/10 \rceil$  is the number of patches and the effective input sequence length for the Transformer." (Section 2.1)
- Patch embedding size: "We flatten each  $16 \times 16$  patch to a 1D patch embedding of size 768 using a linear projection layer." (Section 2.1)
- Variable-length input support: "AST naturally supports variable-length inputs and can be applied to different tasks without any change of architecture." (Section 1. Introduction)
- Input normalization: "We also normalize the input audio spectrogram so that the dataset mean and standard deviation are 0 and 0.5, respectively." (Section 2.2)
- Padding/resizing requirements: Not specified.

---

## 7. Context Window and Attention Structure

- Maximum sequence length (as used in experiments): "An AST that takes 10-second audio input has  $12 \times 100$  patches" (Section 2.2).
- Sequence length fixed or variable: "the length of an audio spectrogram can be variable." (Section 2.2)
- Attention type: Not explicitly specified beyond "a self-attention mechanism" and a standard Transformer encoder. Evidence: "a self-attention mechanism" (Abstract) and "we only use the encoder of the Transformer." (Section 2.1)
- Computational cost management: "increasing the overlap also leads to longer patch sequence inputs to the Transformer, which will quadratically increase the computational overhead." (Section 3.1.3)

---

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: Learnable absolute positional embeddings. Evidence: "we add a trainable positional embedding (also of size 768) to each patch embedding" (Section 2.1).
- Where it is applied: Added to each patch embedding at input. Evidence: "we add a trainable positional embedding (also of size 768) to each patch embedding" (Section 2.1).
- Fixed across experiments or modified: Modified in adaptation and ablated. Evidence: "We propose a cut and bi-linear interpolate method for positional embedding adaptation." (Section 2.2) and "We compare it with a pretrained AST model with a randomly initialized positional embedding." (Section 3.1.3)

---

## 9. Positional Encoding as a Variable

- Treated as a research variable? Yes, in ablation studies of adaptation. Evidence: "Impact of Positional Embedding Adaptation." (Section 3.1.3) and "We compare it with a pretrained AST model with a randomly initialized positional embedding." (Section 3.1.3)
- Multiple positional encodings compared? Yes, adaptation settings are compared. Evidence: "We compare it with a pretrained AST model with a randomly initialized positional embedding." (Section 3.1.3) and "Bi-linear interpolation and nearest-neighbor interpolation do not result in a big difference." (Section 3.1.3)
- Claims PE is not critical? Not stated.

---

## 10. Evidence of Constraint Masking

- Model sizes: "| ViT Base [11]          | 86M      | 0.846    | 0.320    |" (Table 3) "| ViT Large [11]*        | 307M     | 0.851    | 0.330    |" (Table 3) "| DeiT w/ Distill (Used) | 87M      | 0.852    | 0.347    |" (Table 3)
- Dataset sizes: "AudioSet [15] is a collection of over 2 million 10-second audio clips excised from YouTube videos and labeled with the sounds that the clip contains from a set of 527 labels." (Section 3.1.1) "The balanced training, full training, and evaluation set contains 22k, 2M, and 20k samples, respectively." (Section 3.1.1) "The ESC-50 [16] dataset consists of 2,000 5-second environmental audio recordings organized into 50 classes." (Section 3.2) "Speech Commands V2 [17] is a dataset consists of 105,829 1-second recordings of 35 common speech commands." (Section 3.2)
- Performance gains attributed to scaling data or pretraining: "ImageNet pretrained AST noticeably outperforms randomly initialized AST for both balanced and full AudioSet experiments. The performance improvement of ImageNet pretraining is more significant when the training data volume is smaller, demonstrating that ImageNet pretraining can greatly reduce the demand for in-domain audio data for AST." (Section 3.1.3)
- Training tricks contributing to performance: "We train both models with frequency and time masking [29], random noise, and mixup [28] augmentation, a batch size of 128, and the Adam optimizer [32]." (Section 3.2) "As in [8], we also use weight averaging [30] and ensemble [31] strategies to further improve the performance of AST." (Section 3.1.2)
- Architectural/sequence-length tradeoffs: "increasing the overlap also leads to longer patch sequence inputs to the Transformer, which will quadratically increase the computational overhead." (Section 3.1.3)

---

## 11. Architectural Workarounds

- Patch-based tokenization with overlap: "We then split the spectrogram into a sequence of N 16  $\times$  16 patches with an overlap of 6 in both time and frequency dimension" (Section 2.1).
- [CLS] token for classification: "we append a [CLS] token at the beginning of the sequence." (Section 2.1)
- Positional embedding adaptation for transfer: "We propose a cut and bi-linear interpolate method for positional embedding adaptation." (Section 2.2)
- Task-specific head initialization when transferring from ViT: "we abandon the last classification layer of the ViT and reinitialize a new one for AST." (Section 2.2)
- Patch-shape variants to address ordering: "we split the audio spectrogram into  $16 \times 16$  square patches, so the input sequence to the Transformer cannot be in temporal order." (Section 3.1.3) "An alternative way to split the patch is slicing the audio spectrogram into rectangular patches in the temporal order." (Section 3.1.3)

---

## 12. Explicit Limitations and Non-Claims

- Limitation stated: "One disadvantage of the Transformer compared with CNNs is that the Transformer needs more data to train [11]." (Section 2.2)
- Compute tradeoff: "increasing the overlap also leads to longer patch sequence inputs to the Transformer, which will quadratically increase the computational overhead." (Section 3.1.3)
- Future work / explicit non-claims (e.g., open-world learning, unrestrained multi-task learning): Not specified.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: audio-only evaluation across datasets (speech and non-speech), i.e., multiple domains within the same modality (Sections 1, 3.2).
> – Task structure: supervised audio classification tasks (AudioSet, ESC-50, Speech Commands) with dataset-specific label sets (Sections 3.1–3.2).
> – Representation rigidity: fixed log-Mel spectrogram input (128 x 100t) with fixed 16x16 patches and overlap; variable-length sequences via t (Section 2.1).
> – Model sharing vs specialization: shared architecture across tasks, but task-specific training/pretraining regimes (Sections 1, 3.2).
> – Role of positional encoding: learned absolute PE added to patch embeddings, explicitly adapted/ablated for transfer (Sections 2.1, 3.1.3).

---

### 14. Final Classification

**Multi-task, single-domain.** The evaluation covers multiple audio classification tasks/datasets: "we evaluate AST on a variety of audio classification tasks and datasets including AudioSet [15], ESC-50 [16] and Speech Commands [17]." (Section 1. Introduction) The tasks span different audio content but remain within the audio modality: "content varies from speech (Speech Commands) to non-speech (AudioSet and ESC-50)" (Section 3.2).
