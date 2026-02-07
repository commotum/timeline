# LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias (Not specified in the paper)
Source: LVSM- A Large View Synthesis Model with Minimal 3D Inductive Bias.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Novel view synthesis | Sparse posed RGB images; camera intrinsics and extrinsics; target camera pose (Plucker ray embeddings) | 2D (x, y); 0D | Capped (inferred) | Static | Constructed; Direct (inferred) | Novel-view RGB image | 2D (x, y) | Fixed (inferred) |

## Summary
The paper focuses on novel view synthesis from sparse posed RGB images, evaluated on both object-level and scene-level datasets but covering the same core rendering task. Inputs are 2D images with camera pose/intrinsics and a target pose, and outputs are 2D RGB images of the target view. The transformers use dense full self-attention (static by the glossary) and either construct a latent scene representation (encoder-decoder) or directly map inputs to outputs (decoder-only). Input size varies with the number of views and resolution (capped by design/training setups), while each output is a fixed-resolution target image for a given model.

## Evidence
### Task: Novel view synthesis
- "We propose the Large View Synthesis Model (LVSM), a novel transformer-based approach for scalable and generalizable novel view synthesis from sparse-view inputs." (Abstract)
- "Given N sparse input images with known camera poses and intrinsics, denoted as  $\{(\mathbf{I}_i, \mathbf{E}_i, \mathbf{K}_i) | i = 1, \dots, N\}$ , LVSM synthesizes target image  $\mathbf{I}^t$  with novel target camera extrinsics  $\mathbf{E}^t$  and intrinsics  $\mathbf{K}^t$ ." (Sec. 3.1 Overview)
- "We reshape the predicted RGB values back to the 2D patch in  $\mathbb{R}^{p \times p \times 3}$ , and then form the synthesized novel view  $\hat{\mathbf{I}}^t$  by performing the same operation on all target patches independently." (Sec. 3.1 Overview)
- "an encoder-decoder LVSM, which encodes input image tokens into a fixed number of 1D latent tokens, functioning as a fully learned scene representation, and decodes novel-view images from them" (Abstract)
- "a decoder-only LVSM, which directly maps input images to novelview outputs, completely eliminating intermediate scene representations." (Abstract)
- "we adopt dense **full self-attention** across all our encoder and decoder architectures." (Sec. 3.2 Transformer-Based Model Architecture)
- Inference: Marked In Dynamics as Capped (inferred) because input length scales with the number of views and resolution ("We flatten the input tokens into a 1D token sequence, denoted as  $x_1,\ldots,x_{l_x}$ , where  $l_x=NHW/p^2$  is the sequence length of the input image tokens.") and the model is evaluated across varying view counts rather than streaming inputs ("our models, trained on 2 or 4 input views, demonstrate strong zero-shot generalization to an unseen number of views, ranging from a single input to more than 10."). Marked Out Dynamics as Fixed (inferred) because the output is a single target image assembled from fixed-size patches ("We reshape the predicted RGB values back to the 2D patch in  $\mathbb{R}^{p \times p \times 3}$ , and then form the synthesized novel view  $\hat{\mathbf{I}}^t$  by performing the same operation on all target patches independently."). Marked State Dynamic as including Direct (inferred) because the decoder-only architecture "directly maps input images to novelview outputs, completely eliminating intermediate scene representations," which aligns with a direct mapping under the glossary.
