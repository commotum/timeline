# Rotary Masked Autoencoders are Versatile Learners (Not specified in the paper.)
Source: Rotary Masked Autoencoders Are Versatile Learners.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification (irregular multivariate time-series) | irregular multivariate time-series (light curves; multivariate sequences) | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static | Direct | class labels | 0D | Fixed |
| classification (images) | images | 2D (x, y) | Fixed (inferred) | Static | Direct | class labels | 0D | Fixed |
| classification (audio) | audio spectrograms | 2D (x, y) | Fixed (inferred) | Static | Direct | class labels | 0D | Fixed |
| regression (irregular time-series) | irregularly sampled image time-series | 3D (x, y, z) or (x, y, t) | Not specified in the paper. | Static | Direct | sine and cosine of pendulum angle | 1D (t) (inferred) | Not specified in the paper. |
| interpolation (irregular time-series) | irregular time-series trajectories and multivariate clinical records | 1D (t); 2D (x, y) | Fixed; Capped | Static | Direct | interpolated/reconstructed time-series values | 1D (t); 2D (x, y) | Fixed; Capped |
| reconstruction (absolute position) | sequence tokens with continuous positions | 1D (t) | Fixed | Static | Direct | predicted position for each token | 1D (t) | Fixed |

## Summary
The paper supports six explicit tasks across images, audio, and irregular time-series: classification (three modalities), interpolation, regression, and absolute-position reconstruction. The justified dimensional coverage spans 0D outputs, 1D temporal sequences, 2D grids, and 3D spatiotemporal inputs. Dynamics are Fixed or Capped where explicitly supported by dataset/task setup, with regression sequence dynamics left unspecified where the OCR does not state interface bounds. Attention is Static and state is Direct across reported tasks.

## Evidence
### Task: classification (irregular multivariate time-series)
- "We compare RoMAE with state-of-the-art deep learning (DL) models, conducting experiments on the following tasks and modalities: (i) irregularly sampled multi-variate time-series classification, (ii) image classification, (iii) irregularly sampled time-series interpolation and (iv) audio classification." (Section 1 Introduction)
- "The DESC ELAsTICC Challenge is a multi-variate irregular timeseries dataset consisting of  $\sim$ 1.8M simulated light curves and 36 classes of astronomical objects." (Section 5.4 Irregular Time-series Classification)
- "This results in a 2 dimensional positional embedding, where one dimension embeds the time, and the second embeds the channel index." (Section 5.4 Irregular Time-series Classification)
- "Because all variates are present at each time-step, the only irregular dimension is time." (Section 5.4 Irregular Time-series Classification)
- Inference: `1D (t); 2D (x, y) (inferred)` is supported by the paper using both a time-only irregular setup (UEA) and a two-axis time+channel positional setup (ELAsTiCC). `Capped (inferred)` is supported by "variable number of points per sample" with padding and pad masking for ELAsTiCC, indicating variable but bounded sequence handling. (Section D.6 ELAsTiCC Experimental Setup)

### Task: classification (images)
- "We compare RoMAE with state-of-the-art deep learning (DL) models, conducting experiments on the following tasks and modalities: (i) irregularly sampled multi-variate time-series classification, (ii) image classification, (iii) irregularly sampled time-series interpolation and (iv) audio classification." (Section 1 Introduction)
- "To investigate the effect of positional embedding and the learned [CLS] token on RoMAE, we train three versions of RoMAE on Tiny ImageNet [31]; RoPE with the [CLS] token, RoPE without the [CLS] token, and absolute sinusoidal positional embeddings [58] with the [CLS] token." (Section 5.2 Tiny ImageNet)
- Inference: `Fixed (inferred)` input dynamics are supported by fixed image patchification in this setup ("We use a patch size of (16, 16)") and by the stated constant positional regime for images ("If positions stay constant, however, as in images, the overhead becomes negligible."). (Section 5.2 Tiny ImageNet; Section 6 Discussion)

### Task: classification (audio)
- "We compare RoMAE with state-of-the-art deep learning (DL) models, conducting experiments on the following tasks and modalities: (i) irregularly sampled multi-variate time-series classification, (ii) image classification, (iii) irregularly sampled time-series interpolation and (iv) audio classification." (Section 1 Introduction)
- "We chose to test RoMAE's ability to classify audio files, after a self-supervised pre-training on unlabeled audio datasets, inspired by the SSAST pretraining strategy [21]." (Section 5.3 Audio benchmark)
- "For the finetuning audio classification benchmark, we used the ESC-50 dataset [39], consisting of 2000 5-second environmental audio recordings classified into 50 classes." (Section 5.3 Audio benchmark)
- Inference: `Fixed (inferred)` input dynamics are supported by the fixed-duration finetuning benchmark (5-second recordings), and 2D input structure is supported by the spectrogram representation ("This results in a  $128 \times 100~t$  spectrogram."). (Section 5.3 Audio benchmark)

### Task: regression (irregular time-series)
- "Irregular Time-series Regression: Pendulum Dataset The Pendulum dataset [51] is an irregular time-series dataset consisting of irregularly sampled images of a pendulum." (Section 5.4 Irregular Time-series Classification)
- "To embed the images in RoMAE, we use a patch size of (1, 24, 24) for (time, height, width). This corresponds to 1 embedding per time-step/image. RoMAE is trained directly on regression without any pre-training, predicting the sine and cosine of the angle of the pendulum which follows a non-linear dynamical system." (Section 5.4 Irregular Time-series Classification)
- Inference: `1D (t) (inferred)` output dimension is inferred from "1 embedding per time-step/image" and per-sample prediction of sine/cosine angle values, implying temporally indexed regression outputs.

### Task: interpolation (irregular time-series)
- "We evaluate RoMAE on three interpolation tasks with increasing dimensionality and sampling irregularity. (i) Spiral: A 2D synthetic benchmark of 300 noisy Archimedean spirals as in Ref. [12]; (ii) Synthetic: The 50-step univariate task from Ref. [48] and (iii) PhysioNet: 48-hour ICU records containing 41 clinical variables [49]." (Section 5.5 Interpolation)
- "For (i), each spiral is discretized into 75 evenly-spaced time steps." (Section 5.5 Interpolation)
- "For (ii), the interpolation task is between a random subsample including between 3 and 10 points per trajectory." (Section 5.5 Interpolation)
- "Let  $x_t^{(d)} \in \mathbb{R}$  denote the value of feature  $d \in \{1,\dots,41\}$  measured at minute-resolution time step  $t \in \{1,\dots,T\}$   $(T \le 2880)$ ;" (Section D.9 PhysioNet)
- "The two–dimensional positional vector  $p_n$  encodes (i) the normalised time  $t/T \in [0,1]$  and (ii) the feature index d, providing the  $n_{\text{pos}} = 2$  co-ordinates required by RoMAE." (Section D.9 PhysioNet)

### Task: reconstruction (absolute position)
- "To verify the model's ability to reconstruct absolute positional information according to Proposition 4.2, we give the model a sequence of 10 identical values as input. Each embedding is then given a 1D position sampled uniformly between 0 and 50. We then use the same linear head to predict the position for all tokens." (Section 5.1 Reconstructing Absolute Position)
- "We observe a clear difference between the model that uses the [CLS] token and the one that does not. When supplied with the learnable token, RoMAE is able to reconstruct the original absolute position almost perfectly." (Section 5.1 Reconstructing Absolute Position)
