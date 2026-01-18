## 1. Basic Metadata

- Title: "VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text" (Title block)
- Authors: "Hassan Akbari*", "Wei-Hong Chuang", "Liangzhe Yuan", "Shih-Fu Chang", "Boqing Gong", "Rui Qian*", "Yin Cui" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper presents "a framework for learning multimodal representations from unlabeled data using convolution-free Transformer architectures" where "VATT takes raw signals as inputs and extracts multimodal representations that are rich enough to benefit a variety of downstream tasks" (Abstract).

## 3. Tasks Evaluated

Task name: Video action recognition
Task type: Classification
Dataset(s) used: UCF101, HMDB51, Kinetics-400, Kinetics-600, Moments in Time, Kinetics-700
Domain: Video (RGB video frames)
Evidence: "We train VATT end-to-end from scratch using multimodal contrastive losses and evaluate its performance by the downstream tasks of video action recognition" (Abstract); "We use UCF101 [81], HMDB51 [52], Kinetics-400 [14], Kinetics-600 [15], and Moments in Time [61] for video action recognition" (Section 4.1 Experimental Setup); "fine-tuning VATT on the most recent Kinetics-700 dataset results in a top-1 accuracy of 72.7%" (Section 4.2.1 Fine-tuning for video action recognition); "The vision-modality input consists of 3-channel RGB pixels of video frames" (Section 3.1 Tokenization and Positional Encoding).

Task name: Audio event classification
Task type: Classification (multi-label)
Dataset(s) used: ESC50, AudioSet
Domain: Audio waveforms
Evidence: "We use ESC50 [66] and AudioSet [33] for audio event classification" (Section 4.1 Experimental Setup); "We fine-tune VATT's audio Transformer on AudioSet, which benchmarks the task of multi-label audio event classification" (Section 4.2.2 Fine-tuning for audio event classification); "the audio input is in the form of air density amplitudes (waveforms)" (Section 3.1 Tokenization and Positional Encoding).

Task name: Image classification
Task type: Classification
Dataset(s) used: ImageNet
Domain: Images (treated as single-frame video)
Evidence: "Finally, we evaluate the transferability of the vision backbone by fine-tuning it on ImageNet classification [22]" (Section 4.1 Experimental Setup); "performing the image classification task" and "We fine-tune the vision Transformer in VATT-BBS on ImageNet" (Section 4.2.3 Fine-tuning for image classification); "The network sees the input as a single-frame video clip" (Section 4.2.3 Fine-tuning for image classification).

Task name: Text-to-video retrieval (zero-shot)
Task type: Other (retrieval)
Dataset(s) used: YouCook2, MSR-VTT
Domain: Video and text
Evidence: "evaluate its performance by the downstream tasks of ... text-to-video retrieval" (Abstract); "zero-shot text-to-video retrieval on YouCook2 [109] and MSR-VTT [98]" (Section 4.1 Experimental Setup); "We feed video-text pairs to VATT-MBS... Given a text query, we rank the videos based on their similarities to the text" (Section 4.2.4 Zero-shot text-to-video retrieval).

## 4. Domain and Modality Scope

- Single domain? No. Evidence: "downstream tasks of video action recognition, audio event classification, image classification, and text-to-video retrieval" (Abstract).
- Multiple domains within the same modality? Yes. Evidence: "We use UCF101 [81], HMDB51 [52], Kinetics-400 [14], Kinetics-600 [15], and Moments in Time [61] for video action recognition" and "We use ESC50 [66] and AudioSet [33] for audio event classification" (Section 4.1 Experimental Setup).
- Multiple modalities? Yes. Evidence: "Video-Audio-Text Transformer (VATT) takes raw signals as inputs" (Abstract) and "The vision-modality input consists of 3-channel RGB pixels of video frames, the audio input is in the form of air density amplitudes (waveforms), and the text input is a sequence of words" (Section 3.1 Tokenization and Positional Encoding).
- Domain generalization or cross-domain transfer? Yes. Evidence: "Transferring to image classification leads to 78.7% top-1 accuracy on ImageNet ... showing the generalizability of our model despite the domain gap between videos and images" (Abstract).

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Video action recognition | Both modality-specific and modality-agnostic backbones are reported. | Yes. | Not specified; classification uses aggregation token. | "There are two major settings: 1) The backbone Transformers are separate and have specific weights for each modality, and 2) The Transformers share weights" (Section 3 Approach); "three modality-specific variations (LBS, MBS, BBS), and one modality-agnostic (Medium)" and "We fine-tune VATT's vision Transformer on Kinetics-400, Kinetics-600, and Moments in Time" (Section 4.2.1 Fine-tuning for video action recognition); "This will be later used for classification" (Section 3.2 The Transformer Architecture). |
| Audio event classification | Both modality-specific and modality-agnostic backbones are reported. | Yes. | Not specified; classification uses aggregation token. | "We use the final checkpoints of two pre-train settings: one modality-specific (BBS), and one modality-agnostic (Medium)" and "We fine-tune VATT's audio Transformer on AudioSet" (Section 4.2.2 Fine-tuning for audio event classification); "This will be later used for classification" (Section 3.2 The Transformer Architecture). |
| Image classification | Modality-specific backbone used in this experiment (VATT-BBS); shared-backbone option exists in the paper. | Yes. | Not specified; classification uses aggregation token. | "We fine-tune the vision Transformer in VATT-BBS on ImageNet" (Section 4.2.3 Fine-tuning for image classification); "There are two major settings: ... the Transformers share weights" (Section 3 Approach); "This will be later used for classification" (Section 3.2 The Transformer Architecture). |
| Text-to-video retrieval | Both modality-specific and modality-agnostic variants are reported. | No (zero-shot evaluation). | Yes (common space projection heads). | "We feed video-text pairs to VATT-MBS" (Section 4.2.4 Zero-shot text-to-video retrieval) and "Table 4: Zero-shot text-to-video retrieval" (Section 4.2.4); "we define multi-level projections as follows: ... g_{t \to vt} ... g_{v \to vt}" (Section 3.3 Common Space Projection); section title "Zero-shot text-to-video retrieval" (Section 4.2.4). |

## 6. Input and Representation Constraints

- Video input and patching: "The vision-modality input consists of 3-channel RGB pixels of video frames" and "We partition an entire video clip of size T x H x W to a sequence of ceil T/t \cdot ceil H/h \cdot ceil W/w patches, where each patch contains t x h x w x 3 voxels" (Section 3.1 Tokenization and Positional Encoding).
- Audio input and patching: "The raw audio waveform is a 1D input with length T', and we partition it to ceil T'/t' segments each containing t' waveform amplitudes" (Section 3.1 Tokenization and Positional Encoding).
- Text input constraints: "the text input is a sequence of words" and "we first construct a vocabulary of size v ... map each word to a v-dimensional one-hot vector" (Section 3.1 Tokenization and Positional Encoding); "text sequences (capped to 16 tokens)" and "vocabulary size of 2^16" (Section 4.1 Experimental Setup).
- Fixed sampling/resolution in experiments: "We sample 32 frames at 10 fps with a spatial size of 224 x 224" (Section 4.1 Experimental Setup).
- Fixed patch sizes in experiments: "We use patch sizes of 4 x 16 x 16 and 128 for video and raw waveform tokenization, respectively" (Section 4.1 Experimental Setup).
- Normalization and preprocessing: "Both video and audio are normalized between [-1,1]" and "random crop, horizontal flip and color augmentation" (Section 4.1 Experimental Setup).

## 7. Context Window and Attention Structure

- Maximum sequence length: Text is explicitly capped ("text sequences (capped to 16 tokens)") (Section 4.1 Experimental Setup). Maximum sequence length for video/audio is not explicitly stated; token count follows the patching formula "ceil T/t \cdot ceil H/h \cdot ceil W/w" for video and "ceil T'/t'" for audio (Section 3.1 Tokenization and Positional Encoding).
- Fixed or variable sequence length: The formulation uses variable lengths via "ceil T/t" and "ceil H/h" (Section 3.1 Tokenization and Positional Encoding), while experiments sample fixed inputs ("We sample 32 frames ... spatial size of 224 x 224") (Section 4.1 Experimental Setup).
- Attention type: Global self-attention via standard MHA: "We use a standard self-attention [88] as the Multi-Head-Attention (MHA) module" (Section 3.2 The Transformer Architecture).
- Cost-management mechanisms: DropToken reduces token count: "we randomly sample a portion of the tokens and then feed the sampled sequence, not the complete set of tokens, to the Transformer" and "a Transformer's computation complexity is quadratic" (Section 3.1.1 DropToken).

## 8. Positional Encoding (Critical Section)

- Video positional encoding: "To encode the position of these patches, we define a dimension-specific sequence of learnable embeddings as follows: e_{i,j,k} = e_{Temporal_i} + e_{Horizontal_j} + e_{Vertical_k}" (Section 3.1 Tokenization and Positional Encoding).
- Audio positional encoding: "We use ceil T'/t' learnable embeddings to encode the position of each waveform segment" (Section 3.1 Tokenization and Positional Encoding).
- Text positional encoding: "In our text model, we remove the position encoding e_POS and add a learnable relative bias to each attention score of the first layer in the MHA module" (Section 3.2 The Transformer Architecture).
- Where applied: For video/audio, positional encoding is added to inputs: "z_in = [x_AGG; x0 Wp; ...; xN Wp] + e_POS" (Section 3.2 The Transformer Architecture). For text, positional information is injected as a "learnable relative bias" in the first MHA layer (Section 3.2 The Transformer Architecture).
- Fixed vs modified per task/experiment: Modality-specific designs are specified ("each modality has its own positional encoding") (Section 3.1 Tokenization and Positional Encoding); no task-specific modifications or ablations are stated.

## 9. Positional Encoding as a Variable

- Core research variable? Not stated; positional encoding is described as part of the architecture ("each modality has its own positional encoding") (Section 3.1 Tokenization and Positional Encoding).
- Multiple positional encodings compared? Not stated.
- PE choice claimed "not critical" or secondary? Not stated.

## 10. Evidence of Constraint Masking

- Model sizes: "We use 4 network sizes in our experiments ... Medium model (155M parameters) ... BBS (197M), MBS (264M), and LBS (415M)" (Section 4.1 Experimental Setup).
- Dataset sizes: Dataset sizes are not specified; pre-training uses named datasets: "we use a combination of AudioSet [33] and HowTo100M [58] datasets to pre-train VATT" (Section 4.1 Experimental Setup).
- Training scale: "Optimization is performed on totally 500k steps with batch size 2048 (512 in exploration experiments)" (Section 4.1 Experimental Setup).
- Attribution of gains: The paper attributes gains to multimodal self-supervised pre-training, not just scale: "we train a variant from scratch without any pre-training ... The low accuracies verify the efficacy of our pre-training strategy for VATT" (Section 4.2.1 Fine-tuning for video action recognition).
- Training trick to reduce complexity: "DropToken can significantly reduce the pre-training complexity with video and audio modalities" (Section 5 Conclusion and Discussion).

## 11. Architectural Workarounds

- DropToken for compute reduction: "we randomly sample a portion of the tokens and then feed the sampled sequence ... This is crucial for reducing the computational cost" (Section 3.1.1 DropToken).
- Fixed-grid tokenization for raw signals: "We partition an entire video clip ... to a sequence of ... patches" and "we partition [audio] to ceil T'/t' segments" (Section 3.1 Tokenization and Positional Encoding).
- Modality-specific tokenization/PE: "We first define a modality-specific tokenization layer" and "each modality has its own positional encoding" (Section 3.1 Tokenization and Positional Encoding).
- Shared-weight backbone option: "sharing weights among the three modalities" and "a single backbone Transformer applied to any of the modalities" (Abstract; Section 3 Approach).
- Common-space projection heads: "We define a semantically hierarchical common space" and "we define multi-level projections as follows" (Abstract; Section 3.3 Common Space Projection).
- Aggregation token for task heads: "x_AGG is the learnable embedding of a special aggregation token ... used as the aggregated representation for the entire input sequence. This will be later used for classification and common space mapping" (Section 3.2 The Transformer Architecture).

## 12. Explicit Limitations and Non-Claims

- Limitation on multimodal correspondence: "not all videos have organic audio or speech, while our approach depends on meaningful multimodal correspondences" (Section 5 Conclusion and Discussion).
- Noisy/sparse text transcripts: "the text modality currently consists of speech transcripts, which are noisy and sometimes sparse" (Section 5 Conclusion and Discussion).
- Bias risk: "The models could be biased if one applies our approach to the multimodal videos that are not representative enough" (Section 5 Conclusion and Discussion).
- Compute demand: "our method is still demanding in computation" (Section 5 Conclusion and Discussion).
- Explicit non-claims about unrestrained multi-task learning or open-world learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Multi-modal (video, audio, text) with cross-domain transfer to images (Abstract; Section 4.1).
> - Task structure: Multiple fixed downstream tasks ("video action recognition, audio event classification, image classification, and text-to-video retrieval") (Abstract).
> - Representation rigidity: Fixed patching/tokenization and capped text length (Section 3.1; Section 4.1).
> - Model sharing vs specialization: Both modality-specific and shared-backbone variants are used (Section 3 Approach; Section 4.2.1).
> - Role of positional encoding: Modality-specific, learnable embeddings; text uses relative bias instead of e_POS (Section 3.1; Section 3.2).

## 14. Final Classification

**Multi-task, multi-domain (constrained)**

The paper evaluates multiple tasks across modalities: "video action recognition, audio event classification, image classification, and text-to-video retrieval" (Abstract), with explicit datasets for each (Section 4.1). It also claims cross-domain transfer from video to images ("domain gap between videos and images") (Abstract), but the evaluation is on a fixed set of datasets and tasks rather than open-ended or unrestrained multi-task learning.
