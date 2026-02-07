# GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism (Not specified in the paper.)
Source: GPipe- Efficient Training of Giant Neural Networks using Pipeline Parallelism.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | Images | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | Class labels (inferred) | 0D (inferred) | Fixed (inferred) |
| Multilingual neural machine translation | Parallel documents (text) | 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | Translated text sequences (inferred) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper demonstrates GPipe on two tasks: image classification and multilingual neural machine translation. Image experiments use fixed-size 2D inputs (e.g., 480 x 480 images) and produce fixed-class predictions, while translation uses fixed-length text sequences and produces translated text sequences. Attention and state dynamics are not specified for either task.

## Evidence
### Task: Image classification
- "Image Classification: We train a 557-million-parameter AmoebaNet model and attain a top-1 accuracy of 84.4% on ImageNet-2012" (Abstract)
- "We increased the number of channels in an AmoebaNet and scaled the input image size to  $480 \times 480$ ." (Section 4 Image Classification)
- "We changed the number of output units in the last softmax classification layer to the number of classes in the target dataset" (Section 4 Image Classification)
- Inference: Labeled the input as 2D (x, y) and Fixed based on "scaled the input image size to  $480 \times 480$ ."; labeled the output as 0D, Fixed class labels based on "output units in the last softmax classification layer" tied to the "number of classes in the target dataset."

### Task: Multilingual neural machine translation
- "Multilingual Neural Machine Translation: We train a single 6-billion-parameter, 128-layer Transformer model on a corpus spanning over 100 languages" (Abstract)
- "We use a corpus of parallel documents over 102 languages and English" (Section 5 Massive Massively Multilingual Machine Translation)
- "We used a fixed vocabulary size of 32k, sequence length 1024 and batch size 32." (Section 3 Performance Analyses)
- Inference: Labeled inputs/outputs as 1D (t) text sequences and Fixed input dynamics based on "fixed vocabulary size of 32k, sequence length 1024"; output sequence dimensionality is inferred from the machine translation task description and parallel document corpus.
