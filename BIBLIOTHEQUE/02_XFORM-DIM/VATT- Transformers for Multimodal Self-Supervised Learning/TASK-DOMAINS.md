# VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text (Not specified in the paper.)
Source: VATT- Transformers for Multimodal Self-Supervised Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Video action recognition | video clips (RGB frame sequences) | 3D (x, y, t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | action class labels | 0D (inferred) | Fixed (inferred) |
| Audio event classification | audio waveforms | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | audio event labels (multi-label) | 0D (inferred) | Fixed (inferred) |
| Image classification | images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | image class labels | 0D (inferred) | Fixed (inferred) |
| Text-to-video retrieval | text queries; video clips | 1D (t); 3D (x, y, t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | ranked videos | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates four downstream tasks spanning video, audio, image, and cross-modal text-video retrieval. The justified input dimensions range from 1D (t) (audio/text) to 2D (x, y) (images) and 3D (x, y, t) (video), with mostly fixed evaluation interfaces and capped text/query settings where retrieval is involved. Attention is inferred as static for the classification tasks and dynamic for retrieval because the system ranks candidate videos per query. State is inferred as direct for classification mappings and constructed for retrieval due explicit common-space embeddings used for similarity-based ranking.

## Evidence
### Task: Video action recognition
- "We train VATT end-to-end from scratch using multimodal contrastive losses and evaluate its performance by the downstream tasks of video action recognition, audio event classification, image classification, and text-to-video retrieval." (Section Abstract)
- "We fine-tune VATT's vision Transformer on Kinetics-400, Kinetics-600, and Moments in Time, three of the arguably most established large-scale datasets for video action recognition." (Section 4.2.1)
- Inference: In Dimension is labeled 3D (x, y, t) from "The vision-modality input consists of 3-channel RGB pixels of video frames" and "We partition an entire video clip of size  $T \times H \times W$" (Section 3.1). In Dynamics is labeled Fixed from "For video fine-tuning and evaluation, 32 frames with a temporal stride of 2 are sampled at 25 fps (2.56 seconds) with a crop size of  $320\times320$  (with similar video augmentation during pre-training), and we do not drop any tokens." (Section A.2.1). Attention Dynamic and State Dynamic are inferred from "We use a standard self-attention [88] as the Multi-Head-Attention (MHA) module" and "This will be later used for classification and common space mapping." (Section 3.2), supporting Static attention and Direct state for this fixed classification mapping. Out Dimension/Out Dynamics are inferred as 0D/Fixed from classification usage and "final class predictions" (Section A.2.5).

### Task: Audio event classification
- "We fine-tune VATT's audio Transformer on AudioSet, which benchmarks the task of multi-label audio event classification." (Section 4.2.2)
- "The raw audio waveform is a 1D input with length T', and we partition it to  $\lceil T'/t' \rceil$  segments each containing t' waveform amplitudes." (Section 3.1)
- Inference: In Dimension is labeled 1D (t) from the explicit "1D input" statement (Section 3.1). In Dynamics is labeled Fixed from "we employ the duration of 6.4s with 24kHz sampling rate" and "We do not change the input size for audio and text during evaluation" (Sections A.3 and A.2.1). Attention Dynamic and State Dynamic are inferred as Static and Direct from the standard self-attention encoder and classification usage in Section 3.2. Out Dimension/Out Dynamics are inferred as 0D/Fixed because the task is explicit multi-label classification over a fixed label ontology (Section A.1.2 and Section 4.2.2).

### Task: Image classification
- "In this section, we show that our pipeline is capable of transferring the learned knowledge into another domain by performing the image classification task, even though the models are pre-trained in the multimodal video domain." (Section 4.2.3)
- "The network sees the input as a single-frame video clip and performs spatial self-attention." (Section 4.2.3)
- Inference: In Dimension is labeled 2D (x, y) because the conceptual input is images (Section A.1.2), despite implementation as a single-frame clip. In Dynamics is labeled Fixed from "We finetune the pre-trained VATT on ImageNet for 50 epochs with  $384 \times 384$  input resolution" (Section A.3.1). Attention Dynamic and State Dynamic are inferred as Static and Direct from the same fixed-input encoder/classification pathway in Section 3.2. Out Dimension/Out Dynamics are inferred as 0D/Fixed because this is image classification with class predictions.

### Task: Text-to-video retrieval
- "We evaluate the quality of our video-text common space representations by *zero-shot* text-to-video retrieval on two of the most established datasets in this area: YouCook2 [109] and MSR-VTT [98] with 3.1k and 1k video-text pairs, respectively." (Section A.1.2)
- "Given a text query, we rank the videos based on their similarities to the text." (Section 4.2.4)
- Inference: In Dimension is labeled 1D (t); 3D (x, y, t) from text sequences and video clips in Section 3.1. In Dynamics is labeled Capped from "The resulting sequence retains a maximum of 16 words" and fixed clip sampling/evaluation pools (Sections A.2.1 and A.3.3). Attention Dynamic is inferred as Dynamic because runtime retrieval/ranking is query-conditioned (Section 4.2.4). State Dynamic is inferred as Constructed from "extract representations in the  $S_{vt}$  space" and similarity-based ranking over these learned embeddings (Section 4.2.4). Out Dimension/Out Dynamics are inferred as 1D (t)/Capped because output is an ordered ranked list with Recall@10 evaluation (Sections 4.2.4 and A.1.2).
