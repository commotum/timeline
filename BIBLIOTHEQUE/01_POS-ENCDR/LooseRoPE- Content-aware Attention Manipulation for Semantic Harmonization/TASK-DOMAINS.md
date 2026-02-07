# LooseRoPE: Content-aware Attention Manipulation for Semantic Harmonization (Not specified in the paper.)
Source: LooseRoPE- Content-aware Attention Manipulation for Semantic Harmonization.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| semantic harmonization (crop-and-paste image editing) | image with pasted crop; binary mask | 2D (x, y) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | harmonized image | 2D (x, y) | Not specified in the paper. |

## Summary
The paper addresses a single image-editing task: crop-and-paste semantic harmonization, where a crudely edited image and mask are transformed into a harmonized image without text guidance. Inputs and outputs are 2D images (with a 2D mask). The paper does not specify interface size dynamics, so input/output dynamics are not specified. Attention is treated as static and state as constructed (both inferred) based on the fixed image-token attention setup and the use of saliency/VLM-guided internal representations.

## Evidence
### Task: semantic harmonization (crop-and-paste image editing)
- "explicit, prompt-free editing, where the user directly specifies the modification by cropping and pasting an object or sub-object" (Abstract)
- "a task we refer to as semantic harmonization." (Introduction)
- "input image I_in composed of a base image with an additional region crudely pasted on top, along with a binary mask M" (Section 3.2. LooseRoPE)
- "The goal is to produce a harmonized image in which the pasted object or sub-object is seamlessly integrated" (Section 3.2. LooseRoPE)
- Inference: Attention Dynamic marked Static (inferred) because "Output-image queries (within the dotted blue frame) attend to input-image keys" indicates attention operates over fixed input/output tokens (Figure 3).
- Inference: State Dynamic marked Constructed (inferred) because "our method estimates a saliency map and uses it to modulate attention behavior during FLUX Kontext's inference process" and "we leverage a vision-language model (VLM) to automatically steer these parameters during inference." (Section 3.2. LooseRoPE)
