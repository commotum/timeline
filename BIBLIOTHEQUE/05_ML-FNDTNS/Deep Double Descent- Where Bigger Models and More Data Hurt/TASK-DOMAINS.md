# Deep Double Descent: Where Bigger Models and More Data Hurt (Not specified in the paper)
Source: Deep Double Descent- Where Bigger Models and More Data Hurt.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classification (CIFAR-10/100) | Images (CIFAR-10/100 samples) | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | Labels | 0D (inferred) | Fixed (inferred) |
| Translation (IWSLT'14/WMT'14) | Source-language sentences (tokens) | 1D (t) (inferred) | Open (inferred) | Not specified in the paper. | Not specified in the paper. | Target-language sentences (tokens) | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper evaluates supervised image classification on CIFAR-10/100 and neural machine translation on IWSLT'14/WMT'14. The classification experiments imply fixed 2D image inputs and 0D label outputs (inferred from the crop/flip augmentation and label mapping), while translation operates on 1D token sequences with open-length inputs/outputs (inferred from the sentence/token definitions x \in V_{src}^* and y \in V_{tgt}^*). Attention and state dynamics are not explicitly specified for the tasks, beyond noting an encoder-decoder Transformer for translation.

## Evidence
### Task: Classification (CIFAR-10/100)
- "IMAGE CLASSIFICATION: EXPERIMENTAL SETUP" (Section B.2)
- "outputs a classifier  $\mathcal{T}(S)$  mapping data to labels." (Section 2)
- "Model-wise double descent for ResNet18s. Trained on CIFAR-100 and CIFAR-10" (Figure 4 caption)
- "RandomCrop(32, padding=4) and RandomHorizontalFlip." (Section B.2 Data-augmentation)
- Inference: Inferred 2D fixed input dimension from the RandomCrop(32, padding=4) and RandomHorizontalFlip augmentation; inferred 0D fixed output from the classifier mapping to labels.

### Task: Translation (IWSLT'14/WMT'14)
- "language translation (IWSLT'14 German-to-English)." (Figure 3 caption)
- "Transformers on language translation tasks: Multi-head-attention encoder-decoder Transformer model" (Figure 8 caption)
- "x is a sentence in the source language, y is its translation in the target language" (Section B.3)
- "i is the index of the token to be predicted by the model." (Section B.3)
- "x \in V_{src}^*, y \in V_{tat}^*, i \in \{0, \dots, |y|\}" (Section B.3)
- Inference: Inferred 1D (t) input/output and Open input/output dynamics from the token index i and x \in V_{src}^*, y \in V_{tat}^* indicating variable-length sequences.

---

## CSV Output (required)
CSV written to: BIBLIOTHEQUE/05_ML-FNDTNS/Deep Double Descent- Where Bigger Models and More Data Hurt/.TASK-DOMAINS.csv.tmp.c84e29e07a374212aff55d7f71244cba
