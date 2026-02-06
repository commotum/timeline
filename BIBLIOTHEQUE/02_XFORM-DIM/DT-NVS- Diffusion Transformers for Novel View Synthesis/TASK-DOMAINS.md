# DT-NVS: Diffusion Transformers for Novel View Synthesis (Not specified in the paper)
Source: DT-NVS- Diffusion Transformers for Novel View Synthesis.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Generalized novel view synthesis from a single input image | reference image (RGB); camera viewpoint parameters | 2D (x, y); 0D (inferred) | Fixed | Static (inferred) | Constructed | novel-view images | 2D (x, y) | Fixed (inferred) |

## Summary
The paper addresses a single task: generalized novel view synthesis from a single reference image, conditioned on camera viewpoints, producing rendered novel-view images. Inputs and outputs are 2D images (with camera pose parameters as additional 0D inputs), and the setup uses fixed image sizes; output size is treated as fixed in this model. The system constructs an internal 3D radiance-field/scene representation to render views, and it uses self-attention over fixed token sets, implying static attention.

## Evidence
### Task: Generalized novel view synthesis from a single input image
- "We evaluate our approach on the 3D task of generalized novel view synthesis from a single input image" (Abstract)
- "takes the reference image x^r, noisy image z^i_t, and rays from the camera viewpoints c^i and c^r" (Section 4.1 Diffusion in 3D implicit representation)
- "we can render novel views \hat{x}^n from novel viewpoints c^n" (Section 4.3 Rendering)
- "We downsample and center-crop images to 56 × 32 and 32 × 56" (Section 5.1 MVImgNet)
- "predict a radiance field from a single reference image." (Section 1 Introduction)
- Inference: Camera viewpoint parameters are treated as 0D inputs because the paper uses camera viewpoints/poses as parameters alongside images (Section 4.1). Attention Dynamic is marked Static because the decoder uses fixed self-attention over concatenated token sets without runtime selection ("The decoder employs self-attention only, by concatenating feature tokens from the encoder with output tokens" in Section 4.2). Out Dynamics is marked Fixed because outputs are rendered image grids in the same fixed-size setting as the downsampled inputs (Section 5.1).
