# Understanding LSTM Networks (2015)
Source: Understanding LSTM Networks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | movie (video frames) | 3D (x, y, z) or (x, y, t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | event labels at each point in the movie | 1D (t) (inferred) | Open (inferred) |
| speech recognition | speech | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | text tokens (inferred) | 1D (t) (inferred) | Open (inferred) |
| prediction (next-word language modeling) | previous words (tokens) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | next word | 0D (inferred) | Fixed (inferred) |
| translation | source-language tokens (inferred) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | target-language tokens (inferred) | 1D (t) (inferred) | Open (inferred) |
| generation (image captioning) | image | 2D (x, y) (inferred) | Not specified in the paper. | Dynamic | Constructed (inferred) | caption words | 1D (t) (inferred) | Open (inferred) |

## Summary
The OCR text describes recurrent/LSTM use across several sequence-centered tasks: movie event classification, speech recognition, language modeling, and translation, and also mentions attention-based image caption generation. Most supported task interfaces are temporal sequence processing (1D (t)), with one explicit image-conditioned generation example that adds a 2D (x, y) input. Dynamics are mostly inferred as Open because the text frames RNNs as chain-like sequence models without explicit length caps, while next-word prediction is a single-token output step (Fixed output per step). Attention is mostly Static for baseline recurrent processing, with an explicit Dynamic attention case in the image-caption example; state is inferred as Constructed from the cell-state and gating descriptions.

## Evidence
### Task: classification
- "Traditional neural networks can't do this, and it seems like a major shortcoming. For example, imagine you want to classify what kind of event is happening at every point in a movie." (Section Recurrent Neural Networks)
- "LSTMs that this essay will explore. One of the appeals of RNNs is the idea that they might be able to connect previous information to the present task, such as using previous video frames might inform the understanding of the present frame." (Section Recurrent Neural Networks)
- Inference: Movie input was mapped to `3D (x, y, z) or (x, y, t) (inferred)` and output to `1D (t) (inferred)` because the quote specifies classification "at every point in a movie"; `Open (inferred)` follows from recurrent chain processing without explicit length bounds ("allowing information to persist"), `Static (inferred)` reflects no explicit runtime retrieval policy for this baseline example, and `Constructed (inferred)` is supported by explicit cell-state memory and gating.

### Task: speech recognition
- "applying RNNs to a variety of problems: speech recognition, language modeling, translation," (Section Recurrent Neural Networks)
- "An unrolled recurrent neural network. This chain-like nature reveals that recurrent neural networks are intimately related to sequences and lists." (Section Recurrent Neural Networks)
- Inference: `Input`, `Output`, dimensionality, and dynamics are inferred from the task label plus sequence framing (speech as temporal sequence to token sequence, `1D (t) (inferred)`, `Open (inferred)`); `Constructed (inferred)` is supported by explicit cell-state memory ("The key to LSTMs is the cell state") and gating; `Static (inferred)` is used because no explicit dynamic retrieval/selection mechanism is described for this task mention.

### Task: prediction (next-word language modeling)
- "Sometimes, we only need to look at recent information to perform the present task. For example, consider a language model trying to predict the next word based on the previous ones." (Section The Problem of Long-Term Dependencies)
- "state value. In the case of the language model, this is where we'd actually drop the information about the old subject's gender and add the new information, as we decided in the previous steps." (Section Step-by-Step LSTM Walk Through)
- Inference: `1D (t) (inferred)` input and `Open (inferred)` input dynamics are supported by "previous ones" over sequence context; output is `0D (inferred)` and `Fixed (inferred)` because the described operation is single-step next-word prediction; `Constructed (inferred)` is supported by the explicit cell-state example; `Static (inferred)` is used because no separate runtime selection over an external memory/store is described for this baseline language-model example.

### Task: translation
- "applying RNNs to a variety of problems: speech recognition, language modeling, translation," (Section Recurrent Neural Networks)
- "They're the natural architecture of neural network to use for such data." (Section Recurrent Neural Networks)
- Inference: The OCR gives only the task name, so source/target token sequences, `1D (t) (inferred)`, and `Open (inferred)` are inferred from translation as sequence-to-sequence under the stated RNN sequence framing; `Constructed (inferred)` is supported by the LSTM cell-state mechanism; `Static (inferred)` is used because no explicit dynamic retrieval policy is described for this task mention.

### Task: generation (image captioning)
- "The idea is to let every step of an RNN pick information to look at from some larger collection of information." (Section Variants on Long Short Term Memory)
- "For example, if you are using an RNN to create a caption describing an image, it might pick a part of the image to look at for every word it" (Section Variants on Long Short Term Memory)
- Inference: `2D (x, y) (inferred)` input follows from the image reference and `1D (t) (inferred)`/`Open (inferred)` output follows from per-word caption generation; `Dynamic` attention is explicit from "pick information" and "pick a part of the image" at each step; `Constructed (inferred)` follows from recurrent hidden/cell-state sequence generation, while input dynamics remain `Not specified in the paper.`
