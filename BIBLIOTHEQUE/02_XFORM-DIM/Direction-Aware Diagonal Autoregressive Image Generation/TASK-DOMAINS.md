# Direction-Aware Diagonal Autoregressive Image Generation (Not specified in the paper)
Source: Direction-Aware Diagonal Autoregressive Image Generation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| class-conditional image generation | class label token; discrete image tokens (prefix) | 0D; 2D (x, y) | Fixed (inferred) | Static (inferred) | Direct (inferred) | image tokens; images | 2D (x, y) | Fixed (inferred) |

## Summary
The paper covers a single task: class-conditional autoregressive image generation from discrete image tokens, conditioned on class labels. The task operates over 2D image structures (with class conditioning as 0D) and produces 2D images, with experiments centered on fixed 256x256 inputs/outputs. The model uses causal attention over prior tokens (static attention) and a next-token prediction setup that implies direct state usage.

## Evidence
### Task: class-conditional image generation
- "In Tab. 2, we compare DAR with other image generation methods on class-conditional image generation task." (Section 4.2 Main Results)
- "The autoregressive transformer is trained on these sequences using the next-token prediction task, thus acquiring the capability for autoregressive image generation." (Section 3.1 Preliminary)
- "For class-conditional generation, previous methods [47, 64] concatenate the class token at the beginning of the sequence" (Section 3.1 Preliminary)
- "discrete image token sequences produced by visual tokenizers maintain two-dimensional spatial coordinates." (Introduction)
- "During inference, the predicted token sequence is transformed into image pixels by the decoder of the image tokenizer." (Section 3.1 Preliminary)
- Inference: In/Out Dynamics marked Fixed (inferred) because "We train our model on the 256×256 ImageNet-1K [8] dataset" and the tokenizer "it converts  $256 \times 256$  resolution images into  $16 \times 16$  discrete tokens" (Section 4.1 Implementations Details). Attention Dynamic marked Static (inferred) because "Under the constraint of causal attention,  $x_{cur}$  can only attend to the preceding tokens when predicting  $x_{nxt}$ ." (Section 3.2 Diagonal Scanning Order). State Dynamic marked Direct (inferred) because the model uses next-token prediction with "p(x_t|x_1, x_2, ..., x_{t-1}, c)" and no external state is described (Section 3.1 Preliminary).
