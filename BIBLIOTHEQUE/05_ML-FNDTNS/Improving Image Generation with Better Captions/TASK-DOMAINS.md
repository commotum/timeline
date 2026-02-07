# Improving Image Generation with Better Captions (Not specified in the paper.)
Source: Improving Image Generation with Better Captions.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generation (image captioning) | images | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | text captions | 1D (t) (inferred) | Not specified in the paper. |
| generation (text-to-image) | text captions/prompts | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | images | 2D (x, y) (inferred) | Fixed (inferred) |
| generation (caption upsampling) | text captions/prompts | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | expanded/descriptive text captions | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper covers generative image captioning (image-to-text), text-to-image generation (text-to-image), and a caption upsampling step using a language model. The modalities are 2D images and 1D text sequences, with an explicit fixed 256px image size for the diffusion models used in experiments, while most other dynamics are not specified. Attention and state properties are not stated directly, so they are inferred from the fixed conditioning described for the captioner and text-to-image architectures.

## Evidence
### Task: generation (image captioning)
- "We do this by first learning a robust image captioner which produces detailed, accurate descriptions of images." (Section 1 Introduction)
- "To turn this language model into a captioner, you need only to condition on the image." (Section 2.1)
- "the text portion of our corpus can be represented as a sequence, t = [t_1, t_2, \dots, t_n]." (Section 2.1)
- Inference: Labeled input as 2D (x, y) and output as 1D (t), and marked Attention/State as Static/Direct because the captioner conditions on images and predicts token sequences without any described dynamic retrieval or constructed state. (Section 2.1)

### Task: generation (text-to-image)
- "DALL-E 3: a new text-to-image generation system" (Abstract)
- "we used captions from an evaluation dataset to generate 50,000 images from each model." (Section 3.2)
- "The image decoder used in our experiments is a text-conditioned U-Net latent diffusion model." (Appendix A)
- "we train on 256px images, resulting in a model input size of 32x32 latent vectors." (Appendix A)
- Inference: Labeled input as 1D (t) and output as 2D (x, y), marked Attention/State as Static/Direct due to fixed text conditioning, and marked output dynamics as Fixed based on the stated 256px training image size. (Appendix A)

### Task: generation (caption upsampling)
- "we found that GPT-4 will readily \"upsample\" any caption into a highly descriptive one." (Section 3.5)
- "Following is the prompt we give to GPT-4 before feeding it an image caption for \"upsampling\"." (Appendix C)
- "take their short prompts and make them extremely detailed and descriptive." (Appendix C)
- Inference: Labeled input and output dimensions as 1D (t) because the procedure takes text captions and outputs expanded text captions. (Section 3.5, Appendix C)
