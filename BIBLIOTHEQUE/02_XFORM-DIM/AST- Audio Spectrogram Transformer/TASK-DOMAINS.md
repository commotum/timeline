# AST: Audio Spectrogram Transformer (Not specified in the paper.)
Source: AST- Audio Spectrogram Transformer.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Audio event classification | audio spectrograms (log Mel filterbank features) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | labels (audio event labels) | 0D (inferred) | Fixed (inferred) |
| Environmental sound classification | audio spectrograms (log Mel filterbank features) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | class labels | 0D (inferred) | Fixed (inferred) |
| Speech command classification | audio spectrograms (log Mel filterbank features) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | class labels (speech commands) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates AST as an audio classifier on three tasks: weakly-labeled audio event classification (AudioSet), environmental sound classification (ESC-50), and speech command classification (Speech Commands V2). Inputs are 2D log-Mel spectrograms derived from fixed-length audio clips (1 s, 5 s, and 10 s), and outputs are fixed label sets (35, 50, and 527 labels/classes). The architecture is a standard Transformer encoder over the full patch sequence, so attention is treated as static and state as a direct mapping (both inferred).

## Evidence
### Task: Audio event classification
- "we focus on evaluating the AST on AudioSet (Section 3.1) as weakly-labeled audio event classification" (Section 3)
- "AudioSet [15] is a collection of over 2 million 10-second audio clips" (Section 3.1.1)
- "labeled with the sounds that the clip contains from a set of 527 labels." (Section 3.1.1)
- "This results in a  $128 \times 100t$  spectrogram as input to the AST." (Section 2.1)
- "learn a direct mapping from audio spectrograms to corresponding labels." (Abstract)
- Inference: In Dimension is marked 2D and In Dynamics as Fixed based on the spectrogram shape and 10-second clip duration; Out Dimension is 0D and Out Dynamics Fixed based on the label set; Attention is Static and State Direct based on the standard Transformer encoder mapping spectrograms to labels. (Section 2.1; Section 3.1.1; Abstract)

### Task: Environmental sound classification
- "we evaluate AST on a variety of audio classification tasks and datasets including AudioSet [15], ESC-50 [16] and Speech Commands [17]." (Section 1. Introduction)
- "The ESC-50 [16] dataset consists of 2,000 5-second environmental audio recordings organized into 50 classes." (Section 3.2)
- "This results in a  $128 \times 100t$  spectrogram as input to the AST." (Section 2.1)
- "learn a direct mapping from audio spectrograms to corresponding labels." (Abstract)
- Inference: In Dimension is marked 2D and In Dynamics as Fixed based on the spectrogram shape and 5-second clip duration; Out Dimension is 0D and Out Dynamics Fixed based on the class set; Attention is Static and State Direct based on the standard Transformer encoder mapping spectrograms to labels. (Section 2.1; Section 3.2; Abstract)

### Task: Speech command classification
- "we evaluate AST on a variety of audio classification tasks and datasets including AudioSet [15], ESC-50 [16] and Speech Commands [17]." (Section 1. Introduction)
- "Speech Commands V2 [17] is a dataset consists of 105,829 1-second recordings of 35 common speech commands." (Section 3.2)
- "We focus on the 35-class classification task," (Section 3.2)
- "This results in a  $128 \times 100t$  spectrogram as input to the AST." (Section 2.1)
- "learn a direct mapping from audio spectrograms to corresponding labels." (Abstract)
- Inference: In Dimension is marked 2D and In Dynamics as Fixed based on the spectrogram shape and 1-second clip duration; Out Dimension is 0D and Out Dynamics Fixed based on the class set; Attention is Static and State Direct based on the standard Transformer encoder mapping spectrograms to labels. (Section 2.1; Section 3.2; Abstract)
