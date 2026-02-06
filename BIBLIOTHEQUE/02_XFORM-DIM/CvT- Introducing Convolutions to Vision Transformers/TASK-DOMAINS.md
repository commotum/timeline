# CvT: Introducing Convolutions to Vision Transformers (Year not specified)
Source: CvT- Introducing Convolutions to Vision Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image classification | images | 2D (x, y) | Capped (inferred) | Dynamic | Direct (inferred) | class label | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates CvT on image classification across multiple natural-image datasets, using 2D image inputs and producing class-label outputs. It explicitly highlights dynamic attention and a feed-forward classification head. Input sizing is reported at fixed resolutions in experiments while the architecture is described as accommodating variable image resolutions, indicating capped variability rather than open-ended input. No explicit persistent state or external memory is described beyond the classification pathway.

## Evidence
### Task: image classification
- "In this section, we evaluate the CvT model on large-scale image classification datasets and transfer to various down-stream datasets." (Section 4. Experiments)
- "First, the input image (or 2D reshaped token maps) are subjected to the Convolutional Token Embedding layer" (Section 3. Convolutional vision Transformer)
- "Finally, an MLP (i.e. fully connected) Head is utilized upon the classification token of the final stage output to predict the class." (Section 3. Convolutional vision Transformer)
- "dynamic attention, global context" (Abstract)
- Inference: In Dynamics = Capped (inferred) because the paper says it is "capable of accommodating variable resolutions of input images" and reports training at multiple fixed resolutions ("pre-train our models at resolution  $224 \times 224$ , and fine-tune at resolution of  $384 \times 384$ "). State Dynamic = Direct (inferred) because the described pipeline maps an input image through Transformer blocks to a classification token and MLP head without any persistent memory. Out Dimension = 0D (inferred) and Out Dynamics = Fixed (inferred) because the system "predict the class" and datasets specify fixed class counts ("ImageNet dataset, with 1.3M images and 1k classes").
