# Taming Transformers for High-Resolution Image Synthesis (Not specified in the paper)
Source: Taming Transformers for High-Resolution Image Synthesis.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unconditional image synthesis | Image token sequence prefixes (codebook indices) | 2D (x, y) | Capped | Static (inferred) | Direct (inferred) | Images | 2D (x, y) | Open (inferred) |
| Semantic image synthesis | Semantic segmentation masks | 2D (x, y) | Open (inferred) | Static (inferred) | Direct (inferred) | Images | 2D (x, y) | Open (inferred) |
| Structure-to-image synthesis | Depth maps or edge maps | 2D (x, y) | Open (inferred) | Static (inferred) | Direct (inferred) | Images | 2D (x, y) | Open (inferred) |
| Pose-guided synthesis | Pose/keypoint conditions | 2D (x, y) (inferred) | Capped | Static (inferred) | Direct (inferred) | Images | 2D (x, y) | Capped |
| Stochastic superresolution | Low-resolution images | 2D (x, y) | Open (inferred) | Static (inferred) | Direct (inferred) | Upsampled images | 2D (x, y) | Open (inferred) |
| Class-conditional image synthesis | Class label index | 0D | Fixed | Static (inferred) | Direct (inferred) | Images | 2D (x, y) | Capped |
| Image completion | Partial images (half completions) | 2D (x, y) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Completed images | 2D (x, y) | Not specified in the paper. |

## Summary
The paper covers one primary task intent, image generation, across multiple conditional and unconditional settings: semantic synthesis, structure-to-image, pose-guided synthesis, stochastic superresolution, class-conditional synthesis, and image completion. Inputs span both 0D labels and 2D spatial signals (segmentation, depth/edge, pose, low-resolution images), and outputs are consistently 2D images. Interface dynamics are explicitly capped in core training settings (for example, sequence length and crop limits), but the paper also claims arbitrary-ratio/size generation via sliding-window sampling for high-resolution settings. Attention is classified as Static and state as Direct by inference from the autoregressive, decoder-only formulation and fixed-window processing.

## Evidence
### Task: Unconditional image synthesis
- "Results Tab. 1 reports results for unconditional image modeling on ImageNet (IN) [14], Restricted ImageNet (RIN) [65], consisting of a subset of animal classes from ImageNet, LSUN Churches and Towers (LSUN-CT) [79]" (Section 4.1)
- "High-Resolution Synthesis The sliding window approach introduced in Sec. 3.2 enables image synthesis beyond a resolution of  $256 \times 256$  pixels. We evaluate this approach on unconditional image generation on LSUN-CT and FacesHQ" (Section 4.2)
- Inference: Out Dynamics is marked Open because the same section states the sliding-window method "can in principle be used to generate images of arbitrary ratio and size"; Attention Dynamic is Static and State Dynamic is Direct from the autoregressive next-index setup with fixed context limits (Sections 3.2 and 4).

### Task: Semantic image synthesis
- "- (i): **Semantic image synthesis**, where we condition on semantic segmentation masks of ADE20K [83], a webscraped landscapes dataset (S-FLCKR) and COCO-Stuff [6]." (Section 4.2)
- "Figure 5. Samples generated from semantic layouts on S-FLCKR. Sizes from top-to-bottom:  $1280 \times 832$ ,  $1024 \times 416$  and  $1280 \times 240$  pixels." (Figure 5 caption)
- Inference: In/Out Dynamics are marked Open because semantic-layout synthesis is shown at multiple large resolutions and paired with the claim of arbitrary ratio/size generation under sliding-window sampling (Section 4.2).

### Task: Structure-to-image synthesis
- "- (ii): **Structure-to-image**, where we use either depth or edge information to synthesize images from both RIN and IN" (Section 4.2)
- "Figure 6. Applying the sliding attention window approach (Fig. 3) to various conditional image synthesis tasks. Top: Depth-to-image on RIN, ... bottom: Edge-guided synthesis on IN. The resulting images vary between  $368 \times 496$  and  $1024 \times 576$" (Figure 6 caption)
- Inference: In/Out Dynamics are marked Open because this task is explicitly run with sliding-window generation at varying output sizes; Attention Dynamic is Static and State Dynamic is Direct by the same autoregressive fixed-window design (Sections 3.2 and 4.2).

### Task: Pose-guided synthesis
- "- (iii): **Pose-guided synthesis:** ... the same approach as for the previous experiments can be used to build a shape-conditional generative model on the DeepFashion [45] dataset." (Section 4.2)
- "Figure 43. Conditional samples for the pose-guided synthesis model via keypoints on DeepFashion." (Figure 43 caption)
- Inference: Input dimension is marked 2D (x, y) from the keypoint-based conditioning description; Attention Dynamic is Static and State Dynamic is Direct by the same autoregressive conditioned transformer setup (Sections 3.2 and 4.2).

### Task: Stochastic superresolution
- "- (iv): **Stochastic superresolution**, where low-resolution images serve as the conditioning information and are thereby upsampled." (Section 4.2)
- "Figure 36. Additional results for stochastic superresolution with an f = 16 model on IN, using the sliding attention window." (Figure 36 caption)
- Inference: In/Out Dynamics are marked Open because this task is explicitly paired with sliding-window generation and variable high-resolution outputs in conditional-task results; Attention Dynamic is Static and State Dynamic is Direct from the fixed autoregressive formulation (Sections 3.2 and 4.2).

### Task: Class-conditional image synthesis
- "(v): Class-conditional image synthesis: Here, the conditioning information c is a single index describing the class label of interest." (Section 4.2)
- "Class-Conditional Synthesis on ImageNet ... we train a class-conditional ImageNet transformer on  $256 \times 256$  images" (Section 4.4)
- Inference: Attention Dynamic is Static and State Dynamic is Direct from the same decoder-only autoregressive modeling of token sequences conditioned on c (Section 3.2).

### Task: Image completion
- "Figure 4. ... Top row: Completions from unconditional training on ImageNet." (Figure 4 caption)
- "Figure 27. ... we use our f = 16 S-FLCKR model to obtain high-fidelity image completions of the inputs depicted on the left (half completions)." (Figure 27 caption)
- Inference: Attention Dynamic is Static and State Dynamic is Direct by the same autoregressive architecture; In/Out Dynamics are left as "Not specified in the paper." because no explicit bound/open claim is given specifically for completion inputs/outputs.
