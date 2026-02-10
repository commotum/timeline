# Scaling Laws for Autoregressive Generative Modeling (2020)
Source: Scaling Laws for Autoregressive Generative Modeling.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Generative image modeling | Images (RGB pixels / image tokens) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Image tokens / pixels | 2D (x, y) (inferred) | Fixed (inferred) |
| Video modeling / generation | Video frames (VQ tokens over time) | 3D (x, y, t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Video frame tokens / predicted frames | 3D (x, y, t) (inferred) | Fixed (inferred) |
| Text-to-image generation | Text captions (BPE tokens) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Images (RGB pixel tokens) | 2D (x, y) (inferred) | Fixed (inferred) |
| Image-to-text generation | Images (RGB pixel tokens) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Text captions (tokens) | 1D (t) (inferred) | Fixed (inferred) |
| Mathematical problem solving | Math problems as plain-text characters | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Character-level answers | 1D (t) (inferred) | Capped (inferred) |
| Image classification (ImageNet finetuning) | 32x32 images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Class label | 0D (inferred) | Fixed (inferred) |

## Summary
The paper reports autoregressive modeling in image, video, multimodal text-image, and math domains, and also evaluates a downstream image classification task via finetuning. Supported task dimensions span 1D (t), 2D (x, y), and 3D (x, y, t), with a 0D classification output. Most interfaces are fixed in the reported setups (fixed resolutions, fixed frame counts, padded/trimmed caption length), while math is capped by a 512-token context window. Attention is static from fixed dense/sparse transformer attention patterns, and state is treated as direct (inferred) because tasks are implemented as autoregressive prediction/classification without explicit external state construction.

## Evidence
### Task: Generative image modeling
- "We identify empirical scaling laws for the cross-entropy loss in four domains: generative image modeling, video modeling, multimodal image text models, and mathematical problem solving." (Abstract)
- "We study a dataset of approximately  $10^8$  web images [TSF<sup>+</sup>15] scaled to pixel resolutions  $R \times R = 8x8$ , 16x16, and 32x32 represented in raster order using RGB colors, each in the range [0, 255], giving a total of  $3R^2$  tokens per image." (Section 2.1.2 Images)
- "To reduce compute, we use sparse attention patterns [CGRS19], alternating between locally-banded attention and fixed-stride attention in sequential layers, where both the local context length and fixed-stride length are given by the side-length in tokens of the square images." (Section 2.1.2 Images)
- Inference: 2D (x, y), Fixed input/output dynamics, Static attention, and Direct state are inferred from fixed-resolution image-token autoregressive setup and fixed sparse attention pattern; autoregressive formulation is stated as "decoder-only transformer models trained using an autoregressive cross-entropy loss." (Section 2.1)

### Task: Video modeling / generation
- "We identify empirical scaling laws for the cross-entropy loss in four domains: generative image modeling, video modeling, multimodal image text models, and mathematical problem solving." (Abstract)
- "We study a dataset of approximately  $7 \times 10^5$  videos totaling about 100 hours scraped from the web, where each frame is scaled to a pixel resolution of 64x64." (Section 2.1.3 Video)
- "We train on sequences of 16 sequential frames, resulting in a total of 4096 tokens per video." (Section 2.1.3 Video)
- Inference: 3D (x, y, t), Fixed input/output dynamics, Static attention, and Direct state are inferred from fixed frame resolution, fixed 16-frame sequences, fixed sparse attention pattern, and autoregressive token prediction.

### Task: Text-to-image generation
- "Multimodal models are trained to autoregressively predict both image tokens and language tokens in series." (Section 2.1.5 Multimodal Text and Images)
- "We separately study models for text-to-image and image-to-text mappings, as we found poor performance for bidirectional models in preliminary experiments." (Section 2.1.5 Multimodal Text and Images)
- "We use 32x32 images together with a 128-token captions (padded or trimmed as needed), for a total context length of 3200 tokens per image/caption pair." (Section 2.1.5 Multimodal Text and Images)
- Inference: Input 1D (t) text and output 2D (x, y) images are inferred from the explicit text-to-image mapping and caption/image tokenization; Fixed dynamics are inferred from padded/trimmed 128-token captions and fixed 32x32 images; Static attention and Direct state follow the shared autoregressive transformer setup.

### Task: Image-to-text generation
- "Multimodal models are trained to autoregressively predict both image tokens and language tokens in series." (Section 2.1.5 Multimodal Text and Images)
- "We separately study models for text-to-image and image-to-text mappings, as we found poor performance for bidirectional models in preliminary experiments." (Section 2.1.5 Multimodal Text and Images)
- "Similarly, we subtract text losses with and without corresponding images for image-to-text models." (Section 4 Multimodal Models and Information Gain)
- Inference: Input 2D (x, y) images and output 1D (t) text are inferred from the explicit image-to-text direction and text-token loss reporting; Fixed dynamics, Static attention, and Direct state are inferred from fixed caption/image formatting and the same autoregressive transformer interface.

### Task: Mathematical problem solving
- "We identify empirical scaling laws for the cross-entropy loss in four domains: generative image modeling, video modeling, multimodal image text models, and mathematical problem solving." (Abstract)
- "We train and test models using the math problem generator [SGHK19], which generates a variety of problems in algebra, arithmetic, calculus, comparisons, numbers (integer properties), measurement, polynomials, and probability." (Section 2.1.6 Mathematical Problem Solving)
- "A few problem types require interpreting both numbers and strings as sequences of individual characters, so for simplicity we model all questions and responses at the character (byte) level." (Section 2.1.6 Mathematical Problem Solving)
- "The model receives the problems as plain text, and we fill a transformer's 512-token context window with concatenated problems, using a mask so that only the tokens corresponding to answers contribute to the loss." (Section 2.1.6 Mathematical Problem Solving)
- Inference: 1D (t) input/output and Capped dynamics are inferred from character-level text plus a 512-token context window cap; Static attention is inferred from fixed dense attention ("we use dense attention when solving math problems," Section 2.1); Direct state is inferred from autoregressive answer prediction without explicit external state.

### Task: Image classification (ImageNet finetuning)
- "Generative image models can be finetuned for classification." (Section 1 Introduction)
- "We use the scaled-down 32x32 resolution ImageNet [CLH17] and finetune the 32x32 resolution pixel-level generative image models." (Section 3.4 Finetuning on ImageNet at 32x32 Resolution)
- "To turn these models into classifiers, we remove their final embedding matrix and use the mean-pooled (over all pixels) activations of the transformer's final layer as the input to a new single-layer classifier." (Section 3.4 Finetuning on ImageNet at 32x32 Resolution)
- Inference: 2D (x, y) fixed input is inferred from explicit 32x32 image resolution; 0D fixed output is inferred from single-label classification intent; Static attention and Direct state are inferred from the same transformer finetuning setup without runtime retrieval/control.
