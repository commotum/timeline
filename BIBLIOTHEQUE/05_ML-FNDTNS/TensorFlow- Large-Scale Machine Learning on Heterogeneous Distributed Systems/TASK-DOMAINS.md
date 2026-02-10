# TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems (2015)
Source: TensorFlow- Large-Scale Machine Learning on Heterogeneous Distributed Systems.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Deep neural network training | Example records/tensors for model training | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | Updated model parameters (inferred) | Not specified in the paper. | Not specified in the paper. |
| Deep neural network inference | Input tensors/examples for trained models | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | Predictions/computed outputs (inferred) | Not specified in the paper. | Not specified in the paper. |
| Speech recognition | Speech/audio utterances (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Text transcriptions (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Image classification/recognition | Pixel images (e.g., 224 x 224) | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Constructed (inferred) | Class labels | 0D (inferred) | Fixed (inferred) |
| Language modeling | Token sequences (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | Token predictions (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Sequence-to-sequence learning | Source token sequences (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | Target token sequences (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Word embedding training | Text tokens/corpus (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | Word embedding vectors | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper presents TensorFlow as a general framework for both training and inference across multiple machine-learning task areas, with concrete examples spanning speech and vision classification as well as text-sequence modeling. The strongest explicit task specification is image recognition, where the OCR states a fixed 224 x 224 image input and a single label output. Based on named tasks and examples, the covered task spaces include 1D sequence tasks and 2D image tasks with 0D label outputs, while most interface-level dynamics and attention policy details are not explicitly specified per application.

## Evidence
### Task: Deep neural network training
- "The system is flexible and can be used to express a wide variety of algorithms, including training and inference algorithms for deep neural network models, and it has been used for conducting research and for deploying machine learning systems into production across more than a dozen areas of computer science and other fields, including speech recognition, computer vision, robotics, information retrieval, natural language processing, geographic information extraction, and computational drug discovery." (Section Abstract)
- "These clients rely on TensorFlow for research and production, with tasks as diverse as running inference for computer vision models on mobile phones to large-scale training of deep neural networks with hundreds of billions of parameters on hundreds of billions of example records using many hundreds of machines [11, 47, 48, 18, 53, 41]." (Section 1 Introduction)
- Inference: State is marked Constructed and output is marked updated parameters because the paper states "For machine learning applications of TensorFlow, the parameters of the model are typically stored in tensors held in variables, and are updated as part of the *Run* of the training graph for the model." (Section Variables)

### Task: Deep neural network inference
- "The system is flexible and can be used to express a wide variety of algorithms, including training and inference algorithms for deep neural network models, and it has been used for conducting research and for deploying machine learning systems into production across more than a dozen areas of computer science and other fields, including speech recognition, computer vision, robotics, information retrieval, natural language processing, geographic information extraction, and computational drug discovery." (Section Abstract)
- "Running inference on a single image requires 2 billion multiply-add operations." (Section 6 Status and Experience)
- Inference: Output as predictions/computed outputs is inferred from the Run API description: "by the session interface is *Run*, which takes a set of output names that need to be computed, as well as an optional set of tensors to be fed into the graph in place of certain outputs of nodes." (Section Sessions). State is marked Constructed using the Variable/persistent-parameter description in Section Variables.

### Task: Speech recognition
- "The system is flexible and can be used to express a wide variety of algorithms, including training and inference algorithms for deep neural network models, and it has been used for conducting research and for deploying machine learning systems into production across more than a dozen areas of computer science and other fields, including speech recognition, computer vision, robotics, information retrieval, natural language processing, geographic information extraction, and computational drug discovery." (Section Abstract)
- "In addition, often in close collaboration with the Google Brain team, more than 50 teams at Google and other Alphabet companies have deployed deep neural networks using DistBelief in a wide variety of products, including Google Search [11], our advertising products, our speech recognition systems [50, 6, 46], Google Photos [43], Google Maps and StreetView [19], Google Translate [18], YouTube, and many others." (Section 1 Introduction)
- Inference: Input as audio utterances and output as transcribed text (1D (t) to 1D (t)) are inferred from the named task "speech recognition" in the paper.

### Task: Image classification/recognition
- "The examples include models for classifying hand-written digits from the MNIST dataset (the "hello world" of machine learning algorithms) [32], classifying images from the CIFAR-10 dataset [30], doing language modeling using a recurrent LSTM [22] network, training word embedding vectors [35] and more." (Section 6 Status and Experience)
- "This image recognition system classifies  $224 \times 224$  pixel images into one of 1000 labels (e.g., "cheetah", "garbage truck", etc.)." (Section 6 Status and Experience)
- Inference: 2D (x, y) input, 0D output label, and Fixed input/output dynamics are inferred from "224 x 224 pixel images" and "one of 1000 labels." State is marked Constructed because the model is a deep convolutional neural network with learnable parameters in TensorFlow variables.

### Task: Language modeling
- "The examples include models for classifying hand-written digits from the MNIST dataset (the "hello world" of machine learning algorithms) [32], classifying images from the CIFAR-10 dataset [30], doing language modeling using a recurrent LSTM [22] network, training word embedding vectors [35] and more." (Section 6 Status and Experience)
- "Many of the computation graphs for deep neural networks can be quite complex. For example, the computation graph for training a model similar to Google's Inception model [48], a deep convolutional neural net that had the best classification performance in the ImageNet 2014 contest, has over 36,000 nodes in its TensorFlow computation graph, and some deep recurrent LSTM models for language modeling have more than 15,000 nodes." (Section 9.1 Visualization of Computation Graphs)
- Inference: 1D (t) input/output token sequences are inferred from "language modeling." State is marked Constructed because recurrent LSTM models maintain internal recurrent state.

### Task: Sequence-to-sequence learning
- "Figure 8 shows an example of a recurrent, deep LSTM model used for sequence to sequence learning (see [47]), parallelized across three different devices." (Section 7 Model Parallel Training)
- "The approaches in this subsection assume that the model is being trained using stochastic gradient descent (SGD) with relatively modest-sized mini-batches of 100 to 1000 examples." (Section 7 Common Programming Idioms)
- Inference: Source and target token sequences with 1D (t) to 1D (t) are inferred from the named task "sequence to sequence learning." State is marked Constructed based on the recurrent deep LSTM formulation.

### Task: Word embedding training
- "The examples include models for classifying hand-written digits from the MNIST dataset (the "hello world" of machine learning algorithms) [32], classifying images from the CIFAR-10 dataset [30], doing language modeling using a recurrent LSTM [22] network, training word embedding vectors [35] and more." (Section 6 Status and Experience)
- "The system includes detailed documentation, a number of tutorials, and a number of examples demonstrating how to use the system for a variety of different machine learning tasks." (Section 6 Status and Experience)
- Inference: Input as text tokens/corpus and 1D (t) indexing are inferred from the named task "training word embedding vectors." State is marked Constructed because embeddings are learned parameter state.
