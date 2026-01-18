## 1. Basic Metadata

- Title: Rotary Masked Autoencoders are Versatile Learners
- Authors: Uros Zivanovic; Serafina Di Gioia; Andre Scaffidi; Martı́n de los Rios; Gabriella Contardo; Roberto Trotta
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper introduces RoMAE, a Masked Autoencoder extension that applies Rotary Positional Embedding to continuous positions to handle irregular time-series while maintaining performance on images and audio.

---

## 3. Tasks Evaluated

- Task name: Irregular multivariate time-series classification (ELAsTiCC light curves)
  - Task type: Classification
  - Dataset(s) used: DESC ELAsTiCC Challenge
  - Domain: Astronomical light-curve time-series
  - Evidence: "Table 4: Light curve classification results on ELAsTiCC." (Section 5.4 Irregular Time-series Classification); "The DESC ELAsTICC Challenge is a multi-variate irregular timeseries dataset consisting of  $\sim$ 1.8M simulated light curves and 36 classes of astronomical objects." (Section 5.4 Irregular Time-series Classification)

- Task name: Multivariate time-series classification (UEA archive)
  - Task type: Classification
  - Dataset(s) used: UEA Multivariate Time-series Archive datasets
  - Domain: Multivariate time-series (domain not specified)
  - Evidence: "We evaluate RoMAE on a variety of datasets from the UEA Multivariate Time-series Archive [3]." (Section 5.4 Irregular Time-series Classification); "Table 6: Accuracy across various datasets from the UEA Time-series Archive." (Section 5.4 Irregular Time-series Classification)

- Task name: Image classification (Tiny ImageNet)
  - Task type: Classification
  - Dataset(s) used: Tiny ImageNet
  - Domain: Images
  - Evidence: "we train three versions of RoMAE on Tiny ImageNet [31]" (Section 5.2 Tiny ImageNet); "When fine-tuning RoMAE without the [CLS] token, we place the classification head on top of the mean of the output embeddings" (Section 5.2 Tiny ImageNet)

- Task name: Audio classification (ESC-50)
  - Task type: Classification
  - Dataset(s) used: ESC-50 (finetuning); AudioSet-20k and Librispeech (pretraining)
  - Domain: Audio
  - Evidence: "We chose to test RoMAE's ability to classify audio files" (Section 5.3 Audio benchmark); "we thus pretrain RoMAE using two different data sets: AudioSet-20k and the Librispeech dataset." (Section 5.3 Audio benchmark); "For the finetuning audio classification benchmark, we used the ESC-50 dataset [39], consisting of 2000 5-second environmental audio recordings classified into 50 classes." (Section 5.3 Audio benchmark)

- Task name: Irregular time-series regression (Pendulum)
  - Task type: Other (regression)
  - Dataset(s) used: Pendulum dataset
  - Domain: Irregularly sampled image time-series
  - Evidence: "The Pendulum dataset [51] is an irregular time-series dataset consisting of irregularly sampled images of a pendulum." (Section 5.4 Irregular Time-series Classification); "RoMAE is trained directly on regression without any pre-training, predicting the sine and cosine of the angle of the pendulum" (Section 5.4 Irregular Time-series Classification)

- Task name: Irregular time-series interpolation
  - Task type: Reconstruction (interpolation)
  - Dataset(s) used: Spiral, Synthetic, PhysioNet
  - Domain: Synthetic 2D trajectories; synthetic univariate time-series; clinical ICU time-series
  - Evidence: "We evaluate RoMAE on three interpolation tasks with increasing dimensionality and sampling irregularity. (i) Spiral: A 2D synthetic benchmark of 300 noisy Archimedean spirals as in Ref. [12]; (ii) Synthetic: The 50-step univariate task from Ref. [48] and (iii) PhysioNet: 48-hour ICU records containing 41 clinical variables [49]." (Section 5.5 Interpolation)

- Task name: Absolute position reconstruction
  - Task type: Reconstruction
  - Dataset(s) used: Generated synthetic positions
  - Domain: Synthetic sequences with continuous positions
  - Evidence: "To verify the model's ability to reconstruct absolute positional information according to Proposition 4.2, we give the model a sequence of 10 identical values as input. Each embedding is then given a 1D position sampled uniformly between 0 and 50. We then use the same linear head to predict the position for all tokens." (Section 5.1 Reconstructing Absolute Position)

---

## 4. Domain and Modality Scope

- Evaluation scope: Multiple modalities. Evidence: "We showcase RoMAE's performance on a variety of modalities including irregular and multivariate time-series, images, and audio" (Abstract); "conducting experiments on the following tasks and modalities: (i) irregularly sampled multi-variate time-series classification, (ii) image classification, (iii) irregularly sampled time-series interpolation and (iv) audio classification." (Section 1 Introduction)
- Multiple domains within the same modality: Yes, within time-series. Evidence: "The DESC ELAsTICC Challenge is a multi-variate irregular timeseries dataset consisting of  $\sim$ 1.8M simulated light curves and 36 classes of astronomical objects." (Section 5.4 Irregular Time-series Classification); "PhysioNet: 48-hour ICU records containing 41 clinical variables [49]." (Section 5.5 Interpolation); "Spiral: A 2D synthetic benchmark of 300 noisy Archimedean spirals" (Section 5.5 Interpolation)
- Domain generalization or cross-domain transfer: Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Absolute position reconstruction | Not specified. | Not specified. | Yes (linear head). | "We then use the same linear head to predict the position for all tokens." (Section 5.1 Reconstructing Absolute Position) |
| Image classification (Tiny ImageNet) | Not specified. | Yes. | Yes (classification head). | "After pre-training each model for 200 epochs, we fine-tune for another 15." (Section 5.2 Tiny ImageNet); "we place the classification head on top of the mean of the output embeddings" (Section 5.2 Tiny ImageNet) |
| Audio classification (ESC-50) | Not specified. | Yes. | Yes (classification head described). | "we thus pretrain RoMAE using two different data sets: AudioSet-20k and the Librispeech dataset." (Section 5.3 Audio benchmark); "For the finetuning audio classification benchmark, we used the ESC-50 dataset" (Section 5.3 Audio benchmark); "The classification head we use has the same structure as the patch reconstruction head" (Appendix A.2) |
| Irregular time-series classification (ELAsTiCC) | Not specified. | Yes. | Yes (classification head described). | "We train RoMAE-tiny by conducting full pre-training for 200 epochs with a masking ratio of 75%, then fine-tuning for 25 epochs." (Section 5.4 Irregular Time-series Classification); "The classification head we use has the same structure as the patch reconstruction head" (Appendix A.2) |
| Multivariate time-series classification (UEA archive) | No (per-dataset pretraining). | Yes. | Yes (classification head described). | "For each dataset we conduct pre-training for 400 epochs. When fine-tuning, we found it necessary to change hyper-parameters between different datasets." (Section 5.4 Irregular Time-series Classification); "The classification head we use has the same structure as the patch reconstruction head" (Appendix A.2) |
| Irregular time-series regression (Pendulum) | No (trained directly for this task). | No (no pre-training). | Not specified. | "RoMAE is trained directly on regression without any pre-training, predicting the sine and cosine of the angle of the pendulum" (Section 5.4 Irregular Time-series Classification) |
| Irregular time-series interpolation (Spiral/Synthetic/PhysioNet) | Yes (shared pre-training stated). | Not specified. | Not specified. | "one tiny/small RoMAE model, pre-trained once with a generic masked-autoencoder objective and no task-specific architectural tuning, matches or surpasses specialised baselines across three increasingly difficult interpolation datasets." (Section 6 Discussion) |

---

## 6. Input and Representation Constraints

- Input dimensionality: "we consider inputs of the form:  $\mathbf{x} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_D}$  where D is the number of dimensions in  $\mathbf{x}$" (Section 3 Background)
- Fixed patchification and token count: "we define a patch size  $(p_1,\cdots,p_D)$  and divide each dimension into  $N_i=d_i/p_i$  non-overlapping segments" (Section 4 Method); "These are flattened, creating a sequence of patches with length  $k=\\prod_{i=1}^D N_i$  and number of elements per patch  $n_p=\\prod_{i=1}^D p_i$ ." (Section 4 Method)
- Irregular-dimension constraint: "For any *irregular* dimension  $d_i$  in  $\mathbf{x}$ , the corresponding patch size for that dimension  $p_i$  must be equal to 1." (Section 4 Method, Proposition 4.1)
- Dimensionality limits: "In this work we only utilize this process up to D=3." (Section 4 Method)
- Axial RoPE constraints on embedding size: "RoPE requires that embeddings be even and Axial RoPE requires that embeddings be divisible by D" (Section 3.1 Rotary Positional Embeddings)
- Continuous position inputs: "RoMAE also accepts a sequence  $\mathbf{s} = [s_1, \cdots, s_k], \ s_i \in \mathbb{R}^D$ , containing the positional information for each patch." (Section 4.2 Positional Information in RoMAE)
- Example fixed patch sizes: "We use a patch size of (16, 16)" (Section 5.2 Tiny ImageNet); "we split the spectrogram into a sequence of N ( $16 \times 16$ ) patches, where N = 12(100t-16)/10 is the number of patches and the effective input sequence length for the model." (Section 5.3 Audio benchmark); "we use a patch size of (1, 24, 24) for (time, height, width)." (Section 5.4 Irregular Time-series Classification)

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Sequence length fixed or variable: Variable with patchification. Evidence: "creating a sequence of patches with length  $k=\prod_{i=1}^D N_i$" (Section 4 Method); "N = 12(100t-16)/10 is the number of patches and the effective input sequence length for the model." (Section 5.3 Audio benchmark)
- Attention type: Global (standard attention). Evidence: "RoMAE is also not well suited for very long sequences, as it uses standard Attention which has  $O(n^2)$  memory complexity with regards to sequence length." (Section 6 Discussion)
- Mechanisms to manage computational cost: Not specified beyond standard attention; no windowing or sparse attention described. Evidence: "it uses standard Attention which has  $O(n^2)$  memory complexity" (Section 6 Discussion)

---

## 8. Positional Encoding (Critical Section)

- Mechanism: RoPE with continuous positions and axial extensions. Evidence: "We present the Rotary Masked Autoencoder (RoMAE), which utilizes the popular Rotary Positional Embedding (RoPE) method for continuous positions." (Abstract); "we observe that Equation (2) works with any  $m \in \mathbb{R}$ . We make use of this in RoMAE to encode continuous position." (Section 4.2 Positional Information in RoMAE); "To encode multi-dimensional position, we utilize Axial RoPE [17]." (Section 3.1 Rotary Positional Embeddings)
- Variant used: p-RoPE. Evidence: "In this work we make use of p-RoPE [4], a truncated version of RoPE where only the p percent of smallest  $\\theta_i$  values are kept." (Section 3.1 Rotary Positional Embeddings); "We use a value p = 0.75" (Section 3.1 Rotary Positional Embeddings)
- Where applied: "RoPE is applied directly to the queries and keys before they enter SDPA." (Section 3.1 Rotary Positional Embeddings)
- Fixed or modified per task: Compared against absolute sinusoidal embeddings and [CLS] variants. Evidence: "we train three versions of RoMAE on Tiny ImageNet [31]; RoPE with the [CLS] token, RoPE without the [CLS] token, and absolute sinusoidal positional embeddings [58] with the [CLS] token." (Section 5.2 Tiny ImageNet)

---

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption: Core research variable. Evidence: "We investigate how RoPE can be used to embed continuous positions" (Section 1 Introduction, contributions)
- Multiple positional encodings compared: Yes. Evidence: "we train three versions of RoMAE on Tiny ImageNet [31]; RoPE with the [CLS] token, RoPE without the [CLS] token, and absolute sinusoidal positional embeddings [58] with the [CLS] token." (Section 5.2 Tiny ImageNet)
- PE choice claimed as non-critical or secondary: Not stated.

---

## 10. Evidence of Constraint Masking

- Model sizes and scaling: "Throughout the experiments we make use of different sizes of RoMAE: RoMAE-tiny, RoMAE-small, and RoMAE-base" (Experiments); "Table 8: All RoMAE model sizes." (Appendix A.1)
- Model scale attribution: "In the case of RoMAE-tiny, the larger scale of the model also likely plays a role." (Section 5.4 Irregular Time-series Classification)
- Dataset sizes: "The DESC ELAsTICC Challenge is a multi-variate irregular timeseries dataset consisting of  $\sim$ 1.8M simulated light curves and 36 classes of astronomical objects." (Section 5.4 Irregular Time-series Classification); "AudioSet is a 2017 multi-label audio event classification dataset." (Section 5.3 Audio benchmark); "in 2 million 10-second segments of YouTube videos." (Section 5.3 Audio benchmark); "ESC-50 dataset [39], consisting of 2000 5-second environmental audio recordings" (Section 5.3 Audio benchmark); "All the datasets trained on are relatively small, with some being on the order of hundreds of samples." (Section 5.4 Irregular Time-series Classification)
- Data scaling effect: "showing that the size and richness of the pretraining dataset impacts, in a non-negligible way, the performance of that model on the finetuning tasks." (Section 5.3 Audio benchmark)
- Training/architectural factors tied to gains: "A key reason for RoMAE's better performance might be that ATAT does not conduct any pre-training." (Section 5.4 Irregular Time-series Classification); "we attribute to MAE tubelet-masking enforcing long-range reasoning." (Section 6 Discussion)

---

## 11. Architectural Workarounds

- N-dimensional patchification: "we define a patch size  $(p_1,\cdots,p_D)$  and divide each dimension into  $N_i=d_i/p_i$  non-overlapping segments" (Section 4 Method) — used to convert multi-dimensional inputs into token sequences.
- Irregular-dimension handling: "For any *irregular* dimension  $d_i$  in  $\mathbf{x}$ , the corresponding patch size for that dimension  $p_i$  must be equal to 1." (Section 4 Method, Proposition 4.1) — enforces a fixed grid on irregular dimensions.
- Asymmetric encoder/decoder: "RoMAE's structure follows MAE's, using an asymmetric encoder/decoder, with the encoder being much larger than the decoder." (Section 4.1 Overall Structure) — reduces decoder cost.
- [CLS] token for classification: "a learned [CLS] token is optionally appended to the start of the sequence. This token becomes useful during fine-tuning, when an MLP head can be placed on top of it to conduct classification." (Section 4.1 Overall Structure)
- Dimensional index to reduce positional dimensions: "we optionally reserve a dimension in Axial RoPE that is used to store the dimensional index i" and "allows us to reduce the number of positional dimensions from 6 to 2." (Section 4.2 Positional Information in RoMAE)

---

## 12. Explicit Limitations and Non-Claims

- Continuous-position overhead: "RoPE in RoMAE has some additional computational overhead if the positions are different with each forward pass" (Section 6 Discussion, Limitations)
- Long-sequence limitation: "RoMAE is also not well suited for very long sequences, as it uses standard Attention which has  $O(n^2)$  memory complexity with regards to sequence length." (Section 6 Discussion, Limitations)
- Extrapolation limits: "RoMAE's ability to perform on extrapolation tasks is limited" (Section 6 Discussion, Limitations)
- Future work / not tested: "Future work will include building a more robust theoretical understanding of the implications of using RoPE." (Section 7 Conclusion); "we envisage the exploration of the many potential modalities that RoMAE could work with that were not tested here." (Section 7 Conclusion)

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: multiple modalities (irregular time-series, images, audio) but only specific evaluated domains and datasets.
> - Task structure: supervised classification/regression/interpolation with MAE pretraining and dataset-specific training.
> - Representation rigidity: fixed ND patchification; irregular dimensions require patch size 1; Axial RoPE constrains embedding sizes.
> - Model sharing vs specialization: mostly per-task pretraining/fine-tuning, with limited shared pretraining (interpolation); classification uses task heads.
> - Role of positional encoding: central variable (RoPE/p-RoPE/Axial RoPE with comparisons to absolute embeddings and [CLS] variants).

---

### 14. Final Classification

**Multi-task, multi-domain (constrained)**

The paper evaluates across multiple modalities: "irregular and multivariate time-series, images, and audio" (Abstract), and it explicitly targets multiple tasks: "(i) irregularly sampled multi-variate time-series classification, (ii) image classification, (iii) irregularly sampled time-series interpolation and (iv) audio classification." (Section 1 Introduction). The evaluation is constrained to specific datasets and tasks rather than open-ended multi-task or cross-domain transfer, so it is not unrestrained multi-domain learning.
