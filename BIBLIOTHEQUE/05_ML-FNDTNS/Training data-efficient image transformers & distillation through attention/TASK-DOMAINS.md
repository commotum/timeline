# Training data-efficient image transformers & distillation through attention (2021)
Source: Training data-efficient image transformers & distillation through attention.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | Images | 2D (x, y) (inferred) | Fixed | Static (inferred) | Direct (inferred) | Class label | 0D (inferred) | Fixed (inferred) |

## Summary
This paper focuses on image classification, and reports transfer to fine-grained classification benchmarks (CIFAR-10/100, Flowers, Cars, iNaturalist). The model takes images as input at fixed resized resolutions (e.g., 224×224 and 384×384), which supports a 2D input domain with Fixed input dynamics. Outputs are image class labels, giving a 0D output domain with Fixed output dynamics per benchmark label set. The attention is classified as Static and the state as Direct (both inferred from the described feed-forward self-attention classification pipeline without runtime retrieval or persistent constructed state).

## Evidence
### Task: Image classification
- "Recently, neural networks purely based on attention were shown to address image understanding tasks such as image classification." (Abstract)
- "Our models pre-learned on Imagenet are competitive when transferred to different downstream tasks such as fine-grained classification, on several popular public benchmarks: CIFAR-10, CIFAR-100, Oxford-102 flowers, Stanford Cars and iNaturalist-18/19." (Section 1. Introduction)
- "The fixed-size input RGB image is decomposed into a batch of N patches of a fixed size of  $16 \times 16$  pixels ( $N = 14 \times 14$ )." (Section 3. Vision transformer: overview)
- "**The class token** is a trainable vector, appended to the patch tokens before the first layer, that goes through the transformer layers, and is then projected with a linear layer to predict the class." (Section 3. Vision transformer: overview)
- "At test time, both the class or the distillation embeddings produced by the transformer are associated with linear classifiers and able to infer the image label." (Section 4. Distillation through attention)
- Inference: `In Dimension = 2D (x, y)` is inferred from repeated image inputs and fixed image resolutions (e.g., "The fixed-size input RGB image..." and tables using 224/384 image sizes). `Attention Dynamic = Static` is inferred from full self-attention over the given token set ("the attention is in between all the input vectors") with no runtime retrieval/selection mechanism described. `State Dynamic = Direct` is inferred because prediction is produced from the current image tokens/class token in a forward mapping ("only the class vector is used to predict the output"). `Out Dimension = 0D` and `Out Dynamics = Fixed` are inferred from single-label classification outputs ("predict the class", "infer the image label") over fixed dataset label sets.
