# Generative Modeling via Drifting (Not specified in the paper.)
Source: Generative Modeling via Drifting.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generation | Gaussian noise tensors; class labels | 2D (x, y); 0D | Fixed | Static (inferred) | Direct (inferred) | latent images; images | 2D (x, y) | Fixed |
| control | state observations; visual observations | 0D; 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | control actions (inferred) | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper's primary task coverage is one-step generative modeling for images, implemented in both latent space and pixel space. It also evaluates the same drifting-model idea in robotic control settings. The image-generation setup is explicitly fixed-size 2D input/output, while robotics specifies control with state and visual observations but leaves several interface details unstated. Attention and state dynamics are directly justifiable as static/direct only for the image-generation setup based on the described single-pass architecture.

## Evidence
### Task: generation
- "We describe our implementation for image generation on ImageNet (Deng et al., 2009) at resolution 256×256." (Section 4. Implementation for Image Generation)
- "Its input is  $32 \times 32 \times 4$ -dim Gaussian noise  $\epsilon$ , and its output is the generated latent  $\mathbf{x}$  of the same dimension." (Section 4. Implementation for Image Generation)
- "In this case,  $\epsilon$  and  $\mathbf{x}$  are both  $256 \times 256 \times 3$ ." (Section 4. Implementation for Image Generation)
- Inference: `Static (inferred)` and `Direct (inferred)` are inferred from "The mapping f is represented by a single-pass, non-iterative network." and the fixed-shape input/output description above, indicating a reactive fixed-context mapping without runtime retrieval or persistent constructed state. (Section 1. Introduction; Section 4. Implementation for Image Generation)

### Task: control
- "Beyond image generation, we further evaluate our method on robotics control." (Section 5.3. Experiments on Robotic Control)
- "This table involves four single-stage tasks and two multi-stage tasks." (Section 5.3. Experiments on Robotic Control, Table 7 caption)
- "Single-Stag | e Tasks (Si | tate & Visual Obser | vation)" (Section 5.3. Experiments on Robotic Control, Table 7)
- Inference: `0D; 2D (x, y) (inferred)` is inferred from the explicit mention of "State & Visual Observation" (state-like variables plus visual observations). `control actions (inferred)` is inferred from the paper's description of replacing Diffusion Policy with "our one-step Drifting Model" in a robotics control protocol. (Section 5.3. Experiments on Robotic Control)
