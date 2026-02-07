# Recurrent Neural Network Regularization (Not specified in the paper.)
Source: Recurrent Neural Network Regularization.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language modeling (word-level prediction) | word tokens | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Constructed (inferred) | word predictions | 1D (t) (inferred) | Capped (inferred) |
| Speech recognition (acoustic modeling) | acoustic observations (signals) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | phonetic states (frame-level) | 1D (t) (inferred) | Not specified in the paper. |
| Machine translation | source sentence (word sequence) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | translated sentence (word sequence) | 1D (t) (inferred) | Not specified in the paper. |
| Image caption generation | images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | captions (word sequences) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper applies LSTM dropout to four tasks: language modeling, speech recognition (acoustic modeling), machine translation, and image caption generation. The tasks primarily operate over 1D temporal sequences of words or acoustic frames, with one 2D image input; only language modeling explicitly mentions a capped unroll length (35 steps). Attention dynamics are not specified, while the LSTM description supports constructed internal state via memory cells.

## Evidence
### Task: Language modeling (word-level prediction)
- "These tasks include language modeling, speech recognition, image caption generation, and machine translation." (Abstract)
- "We conducted word-level prediction experiments on the Penn Tree Bank (PTB) dataset" (Section 4.1 Language modeling)
- "let  $h_t^0$  be an input word vector at timestep k." (Section 3)
- "We use the activations  $h_t^L$  to predict  $y_t$" (Section 3)
- "Both LSTMs have two layers and are unrolled for 35 steps." (Section 4.1 Language modeling)
- "The \"long term\" memory is stored in a vector of memory cells  $c_t^l \in \mathbb{R}^n$ ." (Section 3.1)
- Inference: In/Out Dimension labeled 1D (t) because inputs/outputs are per-timestep word vectors and predictions; In/Out Dynamics labeled Capped due to "unrolled for 35 steps"; State Dynamic labeled Constructed from LSTM memory cells (Sections 3, 3.1, 4.1).

### Task: Speech recognition (acoustic modeling)
- "These tasks include language modeling, speech recognition, image caption generation, and machine translation." (Abstract)
- "Acoustic modeling is a key component in mapping acoustic signals to sequences of words" (Section 4.2 Speech recognition)
- "where  $s_t$  is the phonetic state at time t and X is the acoustic observation." (Section 4.2 Speech recognition)
- "The \"long term\" memory is stored in a vector of memory cells  $c_t^l \in \mathbb{R}^n$ ." (Section 3.1)
- Inference: In/Out Dimension labeled 1D (t) because acoustic observations and phonetic states are defined at time t; State Dynamic labeled Constructed from LSTM memory cells (Sections 4.2, 3.1).

### Task: Machine translation
- "These tasks include language modeling, speech recognition, image caption generation, and machine translation." (Abstract)
- "We formulate a machine translation problem as a language modelling task" (Section 4.3 Machine translation)
- "assign high probability to a correct translation of a source sentence." (Section 4.3 Machine translation)
- "We compute a translation by approximating the most probable sequence of words" (Section 4.3 Machine translation)
- "The \"long term\" memory is stored in a vector of memory cells  $c_t^l \in \mathbb{R}^n$ ." (Section 3.1)
- Inference: In/Out Dimension labeled 1D (t) because source sentences and translations are sequences of words; State Dynamic labeled Constructed from LSTM memory cells (Sections 4.3, 3.1).

### Task: Image caption generation
- "These tasks include language modeling, speech recognition, image caption generation, and machine translation." (Abstract)
- "We applied the dropout variant to the image caption generation model of Vinyals et al. (2014)." (Section 4.4 Image caption generation)
- "the input image is mapped onto a vector with a highly-accurate pre-trained convolutional neural network" (Section 4.4 Image caption generation)
- "which is converted into a caption with a single-layer LSTM" (Section 4.4 Image caption generation)
- "The \"long term\" memory is stored in a vector of memory cells  $c_t^l \in \mathbb{R}^n$ ." (Section 3.1)
- Inference: Input Dimension labeled 2D (x, y) because the input is an image; Output Dimension labeled 1D (t) because the output is a caption sequence; State Dynamic labeled Constructed from LSTM memory cells (Sections 4.4, 3.1).
