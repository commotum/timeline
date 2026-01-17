## 1. Basic Metadata

- Title: "HEAD-WISE ADAPTIVE ROTARY POSITIONAL ENCODING FOR FINE-GRAINED IMAGE GENERATION" (Title)
- Authors: "Jiaye Li<sup>1</sup>\*, Baoyou Chen<sup>1</sup>\*, Hui Li<sup>1</sup>, Zilong Dong<sup>2</sup>, Jingdong Wang<sup>3</sup>, Siyu Zhu<sup>1,4†</sup>" (Title)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper proposes "HARoPE, a head-wise adaptive extension that inserts a learnable linear transformation parameterized via singular value decomposition (SVD) before the rotary mapping" to address RoPE limitations in "fine-grained image generation" (Abstract).

---

## 3. Tasks Evaluated

- Task name: Image understanding
  - Task type: Classification
  - Dataset(s) used: ImageNet
  - Domain: Image domain (ImageNet)
  - Evidence: "This section evaluates HARoPE across image understanding, class-conditional image generation, and text-to-image generation." (4 EXPERIMENTS) "Image understanding experiments use ImageNet at  $224 \times 224$  with standard resize and center-crop." (4.1 EXPERIMENTAL SETUPS - Dataset) "For image understanding, we report Top-1 accuracy." (4.1 EXPERIMENTAL SETUPS - Metrics)

- Task name: Class-conditional image generation (ImageNet)
  - Task type: Generation
  - Dataset(s) used: ImageNet
  - Domain: Image domain (ImageNet)
  - Evidence: "This section evaluates HARoPE across image understanding, class-conditional image generation, and text-to-image generation." (4 EXPERIMENTS) "For class-conditional image generation, we use DiT-B/2 with a constant learning rate  $1 \times 10^{-4}$ , no weight decay, batch size 256, and EMA with decay 0.9999 for evaluation." (4.1 EXPERIMENTAL SETUPS - Implementation) "For ImageNet generation, we encode images using Stable Diffusion's VAE into  $z \in \mathbb{R}^{H/8 \times W/8 \times 4}$  with  $H \in \{128, 256, 512\}$ ." (4.1 EXPERIMENTAL SETUPS - Dataset) "In class-conditional generation, we adopt ADM's TensorFlow evaluation suite Dhariwal & Nichol (2021) to report FID-50K (Heusel et al., 2017), Inception Score (Salimans et al., 2016), and Precision/Recall (Davis & Goadrich, 2006)." (4.1 EXPERIMENTAL SETUPS - Metrics)

- Task name: Text-to-image generation
  - Task type: Generation
  - Dataset(s) used: BLIP30-60k instruction-tuning set; MS-COCO (train split)
  - Domain: Image domain (text-to-image)
  - Evidence: "This section evaluates HARoPE across image understanding, class-conditional image generation, and text-to-image generation." (4 EXPERIMENTS) "For text-to-image generation, we fine-tune the pretrained FLUX.1-dev model for 4,000 iterations using LoRA (rank 32), AdamW with learning rate  $2 \times 10^{-5}$ , weight decay 0.01, and batch size 64." (4.1 EXPERIMENTAL SETUPS - Implementation) "Text-to-image experiments with the FLUX model use the BLIP30-60k instruction-tuning set of 60k prompt-image pairs." (4.1 EXPERIMENTAL SETUPS - Dataset) "For MMDiT-based text-to-image generation, we utilize the train split of the MS-COCO dataset Lin et al. (2014)." (4.1 EXPERIMENTAL SETUPS - Dataset) "For text-to-image generation, we employ GenEval (Ghosh et al., 2023) and DPG-Bench (Hu et al., 2024) for comprehensive assessment." (4.1 EXPERIMENTAL SETUPS - Metrics)

---

## 4. Domain and Modality Scope

- Single domain vs multiple domains: Evaluation is primarily confined to the image domain. Evidence: "Our evaluation is primarily confined to the image domain due to our computational constraints; the generalizability of the approach to other multi-dimensional data modalities, such as video, audio, or 3D content, remains an open question for empirical validation." (A.1 LIMITATIONS AND FUTURE WORKS)
- Multiple domains within same modality: Not indicated; the paper explicitly states image-domain confinement. Evidence: "Our evaluation is primarily confined to the image domain due to our computational constraints; the generalizability of the approach to other multi-dimensional data modalities, such as video, audio, or 3D content, remains an open question for empirical validation." (A.1 LIMITATIONS AND FUTURE WORKS)
- Multiple modalities: The paper evaluates "text-to-image generation" but does not explicitly characterize modality scope beyond this label. Evidence: "This section evaluates HARoPE across image understanding, class-conditional image generation, and text-to-image generation." (4 EXPERIMENTS)
- Domain generalization or cross-domain transfer: Not claimed. Evidence: "the generalizability of the approach to other multi-dimensional data modalities, such as video, audio, or 3D content, remains an open question for empirical validation." (A.1 LIMITATIONS AND FUTURE WORKS)

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Image understanding | No (separate model) | No (trained from scratch) | Not specified | "For image understanding, we train ViT-B from scratch with AdamW, learning rate  $5 \times 10^{-4}$  and a 5-epoch warmup from  $1 \times 10^{-6}$ , batch size 256, and 300 training epochs." (4.1 EXPERIMENTAL SETUPS - Implementation) |
| Class-conditional ImageNet generation | No (separate model) | Not specified | Not specified | "For class-conditional image generation, we use DiT-B/2 with a constant learning rate  $1 \times 10^{-4}$ , no weight decay, batch size 256, and EMA with decay 0.9999 for evaluation." (4.1 EXPERIMENTAL SETUPS - Implementation) |
| Text-to-image generation (FLUX, MMDiT) | No (separate models) | Yes for FLUX / Not specified for MMDiT | Not specified | "For text-to-image generation, we fine-tune the pretrained FLUX.1-dev model for 4,000 iterations using LoRA (rank 32), AdamW with learning rate  $2 \times 10^{-5}$ , weight decay 0.01, and batch size 64." (4.1 EXPERIMENTAL SETUPS - Implementation)<br>"For MMDiT-based text-to-image generation, we utilize the train split of the MS-COCO dataset Lin et al. (2014)." (4.1 EXPERIMENTAL SETUPS - Dataset) |

---

## 6. Input and Representation Constraints

- Fixed/variable input resolution: "Image understanding experiments use ImageNet at  $224 \times 224$  with standard resize and center-crop." (4.1 EXPERIMENTAL SETUPS - Dataset) "Models are trained on the standard ImageNet-1k resolution of  $224 \times 224$  and tested at progressively larger resolutions." (4.3 ABLATION STUDY - Extrapolation)
- Fixed latent resolution for generation: "For ImageNet generation, we encode images using Stable Diffusion's VAE into  $z \in \mathbb{R}^{H/8 \times W/8 \times 4}$  with  $H \in \{128, 256, 512\}$ ." (4.1 EXPERIMENTAL SETUPS - Dataset)
- High-resolution text-to-image evaluation: "Furthermore, as shown in Table 3, when integrated into the large-scale FLUX model for text-to-image generation at a high resolution of  $1024 \times 1024$ , HARoPE again yields improved performance on both the GenEval and DPG-Bench metrics compared to the original RoPE." (4.3 ABLATION STUDY - Different Image Resolution)
- Dimensionality assumptions: "For 2D positions (x, y), a standard extension partitions the feature dimensions across axes and applies independent rotations:" (3.1 Preliminary: Rotary Position Embeddings) and "For positions  $(x_1, ..., x_p)$  in p dimensions, let  $R_{(x_1, ..., x_p)}$  be the block-diagonal rotary map formed by axis-wise rotations." (3.3 HEAD-WISE ADAPTIVE ROPE)
- Padding/resizing requirements: "standard resize and center-crop." (4.1 EXPERIMENTAL SETUPS - Dataset)
- Fixed patch size: Not specified.
- Fixed number of tokens: Not specified.

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Fixed vs variable sequence length: Not specified.
- Attention type (global/windowed/hierarchical/sparse): Not specified.
- Mechanisms to manage computational cost: "As shown in Table 3, the TFLOPS introduced by the learnable matrices of HARoPE during inference can be very small compared to the entire model." (4.3 ABLATION STUDY - Efficiency and Training Stability)

---

## 8. Positional Encoding (Critical Section)

- Mechanism: RoPE with a head-wise adaptive extension. Evidence: "We propose HARoPE, a head-wise adaptive extension that inserts a learnable linear transformation parameterized via singular value decomposition (SVD) before the rotary mapping." (Abstract) "HARoPE incorporates a lightweight, head-specific linear transformation—parameterized via a singular value decomposition—immediately before the rotary mapping." (3 METHODOLOGY)
- Where applied: "HARoPE inserts, for each attention head h with per-head dimension d, a learnable linear transform  $A_h \in \mathbb{R}^{d \times d}$  immediately before the rotary map." (3.3 HEAD-WISE ADAPTIVE ROPE) "Queries and keys at positions m and n are mapped as" (3.3 HEAD-WISE ADAPTIVE ROPE) "$$\mathbf{q}_h' = R_m A_h \mathbf{q}_h, \qquad \mathbf{k}_h' = R_n A_h \mathbf{k}_h. \tag{7}$$" (3.3 HEAD-WISE ADAPTIVE ROPE)
- Fixed vs modified across experiments: Multiple positional encodings are compared. Evidence: "We compare against strong positional encoding baselines. For image understanding, we include absolute positional embeddings (APE), 2D-RoPE in axial and mixed forms (Heo et al., 2024), STRING (Schenck et al., 2025)/Rethinking RoPE (Liu et al., 2025), and HAROPE. For class-conditional generation on ImageNet, we evaluate APE, Vanilla RoPE, 2D-RoPE (Axial), VideoRoPE (Wei et al., 2025), STRING/Rethinking RoPE, and HAROPE. For text-to-image generation, we directly replace RoPE in FLUX with HAROPE and APE in MMDiT for a controlled comparison." (4.1 EXPERIMENTAL SETUPS - Baselines)

---

## 9. Positional Encoding as a Variable

- Core research variable vs fixed assumption: Treated as a core research variable. Evidence: "We propose HARoPE, a head-wise adaptive extension that inserts a learnable linear transformation parameterized via singular value decomposition (SVD) before the rotary mapping." (Abstract) and "We compare against strong positional encoding baselines." (4.1 EXPERIMENTAL SETUPS - Baselines)
- Multiple positional encodings compared: Yes. Evidence: "We compare against strong positional encoding baselines. For image understanding, we include absolute positional embeddings (APE), 2D-RoPE in axial and mixed forms (Heo et al., 2024), STRING (Schenck et al., 2025)/Rethinking RoPE (Liu et al., 2025), and HAROPE." (4.1 EXPERIMENTAL SETUPS - Baselines)
- PE choice claimed “not critical” or secondary: Not claimed.

---

## 10. Evidence of Constraint Masking

- Model size(s): Model families are specified but parameter counts are not. Evidence: "For image understanding, we train ViT-B from scratch with AdamW, learning rate  $5 \times 10^{-4}$  and a 5-epoch warmup from  $1 \times 10^{-6}$ , batch size 256, and 300 training epochs." and "For class-conditional image generation, we use DiT-B/2 with a constant learning rate  $1 \times 10^{-4}$ , no weight decay, batch size 256, and EMA with decay 0.9999 for evaluation." and "For text-to-image generation, we fine-tune the pretrained FLUX.1-dev model for 4,000 iterations using LoRA (rank 32), AdamW with learning rate  $2 \times 10^{-5}$ , weight decay 0.01, and batch size 64." (4.1 EXPERIMENTAL SETUPS - Implementation)
- Dataset size(s): Only one explicit size is stated. Evidence: "Text-to-image experiments with the FLUX model use the BLIP30-60k instruction-tuning set of 60k prompt-image pairs." (4.1 EXPERIMENTAL SETUPS - Dataset)
- Attribution of gains: Performance improvements are attributed to the positional encoding modification rather than scaling. Evidence: "This lightweight modification enables dynamic frequency reallocation, semantic alignment of rotary planes, and head-specific positional receptive fields while rigorously preserving RoPE's relative-position property." and "Extensive experiments on class-conditional ImageNet and text-to-image generation (Flux and MMDiT) demonstrate that HARoPE consistently improves performance over strong RoPE baselines and other extensions." (Abstract)
- Training tricks: LoRA is used for FLUX fine-tuning. Evidence: "For text-to-image generation, we fine-tune the pretrained FLUX.1-dev model for 4,000 iterations using LoRA (rank 32), AdamW with learning rate  $2 \times 10^{-5}$ , weight decay 0.01, and batch size 64." (4.1 EXPERIMENTAL SETUPS - Implementation)

---

## 11. Architectural Workarounds

- Head-wise adaptive linear transform before RoPE: "We propose HARoPE, a head-wise adaptive extension that inserts a learnable linear transformation parameterized via singular value decomposition (SVD) before the rotary mapping." (Abstract) Purpose described as: "This lightweight modification enables dynamic frequency reallocation, semantic alignment of rotary planes, and head-specific positional receptive fields while rigorously preserving RoPE's relative-position property." (Abstract)
- Head-specific adaptation per attention head: "Moreover, endowing each attention head with an independent SVD equips the model with specialized positional receptive fields, promoting complementary multi-scale behaviors." (1 Introduction)
- Latent-space representation for ImageNet generation: "For ImageNet generation, we encode images using Stable Diffusion's VAE into  $z \in \mathbb{R}^{H/8 \times W/8 \times 4}$  with  $H \in \{128, 256, 512\}$ ." (4.1 EXPERIMENTAL SETUPS - Dataset)
- Low-rank adaptation for text-to-image fine-tuning: "For text-to-image generation, we fine-tune the pretrained FLUX.1-dev model for 4,000 iterations using LoRA (rank 32), AdamW with learning rate  $2 \times 10^{-5}$ , weight decay 0.01, and batch size 64." (4.1 EXPERIMENTAL SETUPS - Implementation)
- Windowed attention, hierarchical stages, token pooling/merging, or task-specific heads: Not mentioned.

---

## 12. Explicit Limitations and Non-Claims

- Limitations: "Our evaluation is primarily confined to the image domain due to our computational constraints; the generalizability of the approach to other multi-dimensional data modalities, such as video, audio, or 3D content, remains an open question for empirical validation." (A.1 LIMITATIONS AND FUTURE WORKS)
- Additional limitation: "Another consideration is the static nature of the learned transformation matrices, which are fixed after training. Although the head-wise specialization is beneficial, the adaptation process is not input-conditional. Exploring dynamic transformations that can adapt based on input content or evolve during inference could further enhance the flexibility and performance of the positional encoding mechanism." (A.1 LIMITATIONS AND FUTURE WORKS)
- Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: "Our evaluation is primarily confined to the image domain due to our computational constraints; the generalizability of the approach to other multi-dimensional data modalities, such as video, audio, or 3D content, remains an open question for empirical validation." (A.1 LIMITATIONS AND FUTURE WORKS).
> – Task structure: "This section evaluates HARoPE across image understanding, class-conditional image generation, and text-to-image generation." (4 EXPERIMENTS).
> – Representation rigidity: "Image understanding experiments use ImageNet at  $224 \times 224$  with standard resize and center-crop." and "For ImageNet generation, we encode images using Stable Diffusion's VAE into  $z \in \mathbb{R}^{H/8 \times W/8 \times 4}$  with  $H \in \{128, 256, 512\}$ ." (4.1 EXPERIMENTAL SETUPS - Dataset).
> – Model sharing vs specialization: Separate models are used per task ("For image understanding, we train ViT-B from scratch with AdamW, learning rate  $5 \times 10^{-4}$  and a 5-epoch warmup from  $1 \times 10^{-6}$ , batch size 256, and 300 training epochs." / "For class-conditional image generation, we use DiT-B/2 with a constant learning rate  $1 \times 10^{-4}$ , no weight decay, batch size 256, and EMA with decay 0.9999 for evaluation." / "For text-to-image generation, we fine-tune the pretrained FLUX.1-dev model for 4,000 iterations using LoRA (rank 32), AdamW with learning rate  $2 \times 10^{-5}$ , weight decay 0.01, and batch size 64.") while within-model specialization is head-wise ("Moreover, endowing each attention head with an independent SVD equips the model with specialized positional receptive fields, promoting complementary multi-scale behaviors."). (4.1 EXPERIMENTAL SETUPS - Implementation; 1 Introduction).
> – Role of positional encoding: Positional encoding is the main experimental variable ("We propose HARoPE, a head-wise adaptive extension that inserts a learnable linear transformation parameterized via singular value decomposition (SVD) before the rotary mapping." (Abstract) and "We compare against strong positional encoding baselines." (4.1 EXPERIMENTAL SETUPS - Baselines)).

---

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple tasks—"image understanding, class-conditional image generation, and text-to-image generation" (4 EXPERIMENTS)—but explicitly states that "our evaluation is primarily confined to the image domain due to our computational constraints; the generalizability of the approach to other multi-dimensional data modalities, such as video, audio, or 3D content, remains an open question for empirical validation." (A.1 LIMITATIONS AND FUTURE WORKS). It does not claim cross-domain transfer, and generalizability to other modalities is left as an open question.
