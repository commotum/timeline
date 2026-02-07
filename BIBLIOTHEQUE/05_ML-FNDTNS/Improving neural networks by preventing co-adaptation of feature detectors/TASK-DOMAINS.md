# Improving neural networks by preventing co-adaptation of feature detectors (Not specified in the paper.)
Source: Improving neural networks by preventing co-adaptation of feature detectors.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification (handwritten digit images) | handwritten digit images (28x28) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | digit class label | 0D (inferred) | Fixed (inferred) |
| classification (HMM state per frame) | 21 adjacent acoustic frames (window) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | HMM state label / class probabilities | 0D (inferred) | Fixed (inferred) |
| sequence inference (HMM state sequence) | per-frame class probabilities (sequence of frames) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | sequence of HMM states | 1D (t) (inferred) | Not specified in the paper. |
| object recognition (image classification) | images (CIFAR-10 32x32 color; ImageNet resized to 256x256) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | object class label | 0D (inferred) | Fixed (inferred) |
| classification (news topic documents) | documents (word-count vectors of 2000 non-stop words) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | topic class label | 0D (inferred) | Fixed (inferred) |

## Summary
The paper applies dropout to supervised classification tasks across images (MNIST digits, CIFAR-10, ImageNet), speech (TIMIT frame-level HMM state classification), and text documents (Reuters), and it also reports sequence-level speech recognition via Viterbi decoding. Inputs are 2D image grids or 1D sequences/vectors with fixed sizes for the classification tasks; sequence length for recognition is not specified. Attention and state dynamics are not explicitly defined, but the feedforward classifiers imply static attention and direct state, while the Viterbi decoding implies constructed state for sequence inference.

## Evidence
### Task: classification (handwritten digit images)
- "The MNIST dataset consists of  $28 \times 28$  digit images" (Section A.1 Details for dropout training)
- "The objective is to classify the digit images into their correct digit class." (Section A.1 Details for dropout training)
- "A feedforward, artificial neural network uses layers of non-linear \"hidden\" units between its inputs and its outputs." (Main text)
- Inference: 2D and Fixed input inferred from "28 x 28 digit images"; 0D/Fixed output inferred from "digit class"; Static/Direct inferred from the feedforward input-to-output mapping described in the main text.

### Task: classification (HMM state per frame)
- "map a short sequence of frames into a probability distribution over HMM states" (Main text, TIMIT paragraph)
- "The input to the net is 21 adjacent frames with an advance of 10ms per frame." (Main text, TIMIT paragraph)
- "the central frame of a window is classified as belonging to the HMM state" (Main text, TIMIT paragraph)
- Inference: 1D and Fixed input inferred from the 21-frame window; 0D/Fixed output inferred from per-frame HMM state classification; Static/Direct inferred from the feedforward mapping of the fixed window to a state distribution.

### Task: sequence inference (HMM state sequence)
- "the class probabilities that the neural network outputs for each frame are given to a decoder" (Main text, TIMIT paragraph)
- "runs the Viterbi algorithm to infer the single best sequence of HMM states." (Main text, TIMIT paragraph)
- Inference: 1D input/output inferred from per-frame probabilities and the "sequence of HMM states"; Constructed state inferred from use of the Viterbi algorithm for sequence inference.

### Task: object recognition (image classification)
- "CIFAR-10 is a benchmark task for object recognition." (Main text, CIFAR-10 paragraph)
- "It uses 32x32 downsampled color images of 10 different object classes" (Main text, CIFAR-10 paragraph)
- "a neural network with three convolutional hidden layers interleaved with three \"max-pooling\" layers" (Main text, CIFAR-10 paragraph)
- "ImageNet is an extremely challenging object recognition dataset consisting of thousands of high-resolution images of thousands of classes of object" (Main text, ImageNet paragraph)
- "For our experiments we resized all images to  $256 \\times 256$  pixels." (Section E ImageNet)
- Inference: 2D and Fixed input inferred from the fixed-size image descriptions; 0D/Fixed output inferred from object class labels; Static/Direct inferred from the feedforward CNN classification setup described for CIFAR-10/ImageNet.

### Task: classification (news topic documents)
- "The Reuters dataset contains documents that have been labeled with a hierarchy of classes." (Main text, Reuters paragraph)
- "Each document was represented by a vector of counts for 2000 common non-stop words" (Main text, Reuters paragraph)
- "We created training and test sets each containing 201,369 documents from 50 mutually exclusive classes." (Main text, Reuters paragraph)
- "A feedforward neural network with 2 fully connected layers of 2000 hidden units" (Main text, Reuters paragraph)
- Inference: 1D and Fixed input inferred from the 2000-word count vector; 0D/Fixed output inferred from mutually exclusive classes; Static/Direct inferred from feedforward document classification setup.
