# ViViT: A Video Vision Transformer (Year not specified in the paper)
Source: ViViT- A Video Vision Transformer.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Video classification | Video clips | 3D (x, y, t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Class label | 0D (inferred) | Fixed (inferred) |
| Action recognition (verb+noun, action metric) | Egocentric video clips | 3D (x, y, t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Verb label + noun label; action pair | 0D (inferred) | Fixed (inferred) |

## Summary
The paper primarily covers video classification across multiple datasets, and also reports a distinct Epic Kitchens action-recognition protocol that predicts verb and noun labels and evaluates the derived action. Inputs are video clips (spatiotemporal), supporting 3D (x, y, t) task domains, while outputs are discrete labels (0D). The described setup uses bounded clip/view processing and fixed clip-to-label prediction, supporting Capped input dynamics, Static attention behavior, and Direct state.

## Evidence
### Task: Video classification
- "We present pure-transformer based models for video classification, drawing upon the recent success of such models in image classification." (Abstract)
- "We evaluate the performance of our proposed models on a diverse set of video classification datasets:" (Section 4.1. Experimental Setup, Datasets)
- "The input to our network is a video clip of 32 frames using a stride of 2, unless otherwise mentioned, similar to [20, 19]." (Section 4.1. Experimental Setup, Inference)
- "Finally, a linear classifier is used to classify the encoded input based on  $z_{cls}^L \in \mathbb{R}^d$ , if it was prepended to the input, or a global average pooling of all the tokens,  $\mathbf{z}^L$ , otherwise." (Section 3.1. Overview of Vision Transformers (ViT))
- Inference: In Dimension is marked as 3D (x, y, t) because the task input is explicitly video ("mapping a video  $\mathbf{V} \in \mathbb{R}^{T \times H \times W \times C}$ " in Section 3.2). In Dynamics is marked Capped from the finite clip/view interface ("video clip of 32 frames ... unless otherwise mentioned" in Section 4.1 and fixed numbers of views in Section 4.1). Attention Dynamic is marked Static because attention is computed over the provided tokenized clip rather than runtime-selected external context ("each transformer layer models all pairwise interactions between all spatio-temporal tokens" in Section 3.3). State Dynamic is marked Direct because the model is described as direct encoded-input-to-classifier prediction (Section 3.1).

### Task: Action recognition (verb+noun, action metric)
- "We report results following the standard \"action recognition\" protocol." (Section 4.1. Experimental Setup, Datasets)
- "Here, each video is labelled with a \"verb\" and a \"noun\" and we therefore predict both categories using a single network with two \"heads\"." (Section 4.1. Experimental Setup, Datasets)
- "The top-scoring verb and action pair predicted by the network form an \"action\", and action accuracy is the primary metric." (Section 4.1. Experimental Setup, Datasets)
- Inference: In Dimension is marked as 3D (x, y, t) because Epic Kitchens inputs are videos and the model input is a video clip (Sections 4.1 and 3.2). In Dynamics is marked Capped from the same finite clip/view setup in Section 4.1. Attention Dynamic is marked Static and State Dynamic is marked Direct for the same architecture-level reasons as the video-classification row (Sections 3.1 and 3.3).
