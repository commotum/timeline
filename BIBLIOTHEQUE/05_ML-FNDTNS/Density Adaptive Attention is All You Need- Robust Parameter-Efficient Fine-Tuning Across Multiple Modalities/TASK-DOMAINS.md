# Density Adaptive Attention is All You Need: Robust Parameter-Efficient Fine-Tuning Across Multiple Modalities (Not specified in the paper.)
Source: Density Adaptive Attention is All You Need- Robust Parameter-Efficient Fine-Tuning Across Multiple Modalities.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Emotion recognition (speech) | speech audio clips | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | emotion class label (inferred) | 0D (inferred) | Fixed (inferred) |
| Text classification | text tokens (title and description) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | class label (inferred) | 0D (inferred) | Fixed (inferred) |
| Image classification | images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | class label (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates DAAM/DAT on three downstream classification tasks spanning speech emotion recognition (IEMOCAP), text classification (AG News), and image classification (CIFAR-100). Inputs cover 1D temporal sequences (audio and tokens) and 2D spatial images, with capped or fixed input sizes based on the described preprocessing (max 5-second clips, 4096-token context, 224x224 images). Attention is described within a fixed-input transformer pipeline, so attention dynamics are inferred as static, and the use of contextualized representations implies constructed state.

## Evidence
### Task: Emotion recognition (speech)
- "tasks, including emotion recognition in speech, image classification, and text classification" (Abstract)
- "We focus on the emotion categories neutral, happiness (merging happiness and excited), anger, and sadness." (Section 1.3 Datasets)
- "audio files are split to a maximum of 5 second clips." (Section 1.4 Encoder and Decoder Models)
- "The attention mechanism of the module then produces a new, contextualized representation  $C \\in \\mathbb{R}^{N \\times d}$  for the input sequence." (Section 1.4 Encoder and Decoder Models)
- Inference: Inferred 1D (t) input and Capped dynamics from the 5-second clip constraint; inferred output as an emotion class label (0D, Fixed) from the listed emotion categories; inferred Static attention due to fixed-length inputs; inferred Constructed state from the contextualized representation. (Sections 1.3 and 1.4)

### Task: Text classification
- "tasks, including emotion recognition in speech, image classification, and text classification" (Abstract)
- "Our dataset construction focuses solely on the title and description fields of these articles." (Section 1.3 Datasets)
- "text is tokenized with maximum context length of 4096 during both training and evaluation." (Section 1.4 Encoder and Decoder Models)
- "The attention mechanism of the module then produces a new, contextualized representation  $C \\in \\mathbb{R}^{N \\times d}$  for the input sequence." (Section 1.4 Encoder and Decoder Models)
- Inference: Inferred 1D (t) input and Capped dynamics from tokenized text with a maximum context length; inferred output as a class label (0D, Fixed) from the classification task framing; inferred Static attention due to fixed-length inputs; inferred Constructed state from the contextualized representation. (Sections 1.3 and 1.4)

### Task: Image classification
- "tasks, including emotion recognition in speech, image classification, and text classification" (Abstract)
- "For the image classification downstream task, images are resized to  $224 \times 224$  during both training and evaluation." (Section 1.4 Encoder and Decoder Models)
- "The attention mechanism of the module then produces a new, contextualized representation  $C \\in \\mathbb{R}^{N \\times d}$  for the input sequence." (Section 1.4 Encoder and Decoder Models)
- Inference: Inferred 2D (x, y) input and Fixed dynamics from the 224x224 resizing; inferred output as a class label (0D, Fixed) from the classification task framing; inferred Static attention due to fixed-size inputs; inferred Constructed state from the contextualized representation. (Section 1.4)
