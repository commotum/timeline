# MIND OVER BODY: ADAPTIVE THINKING USING DYNAMIC COMPUTATION (Not specified in the paper)
Source: Adaptive Thinking Using Dynamic Computation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | images | 2D (x, y) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | class label (inferred) | 0D (inferred) | Fixed (inferred) |
| Motion direction identification (random dot) | two images (original and shifted dot displays) | 3D (x, y, t) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | direction label (left/right/up/down) (inferred) | 0D (inferred) | Fixed (inferred) |
| Language modeling | tokens | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Question answering | question + passage tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | answer text/tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Summarization | document text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | summary text/tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates MIND on vision classification tasks (e.g., CIFAR/ImageNet), a random-dot motion direction task, and NLP tasks covering language modeling, question answering, and summarization. Inputs span 2D images, two-frame motion inputs (spatiotemporal), and 1D token sequences; outputs are labels or text sequences, with sequence lengths capped by the transformer configuration. The introspection mechanism dynamically adjusts computation based on intermediate activations, supporting Dynamic attention and Constructed state across tasks (inferred from the architecture description).

## Evidence
### Task: Image classification
- "in image classification tasks." (Section 4.3 Experiments on Vision Tasks)
- "We evaluated our models on CIFAR-100 (Krizhevsky, 2009) and ImageNet (Deng et al., 2009) datasets models using Top-1 and Top-5 accuracy metrics." (Section 4.3 Experiments on Vision Tasks)
- "CIFAR-100 consists of  $60,000\ 32\times 32$  images in 100 classes, while ImageNet has 1.28M images in 1,000 classes." (Vision Experiments)
- Inference: Inferred 2D fixed image inputs and 0D label outputs from "32\times 32" images and "Top-1 and Top-5 accuracy metrics"; inferred Dynamic attention and Constructed state because the introspection network "assesses the complexity of the input and the current activation states, dynamically adjusting the computational graph" and is "responsible for analyzing intermediate activations." (Sections Vision Experiments, 3.1)

### Task: Motion direction identification (random dot)
- "The animal must identify the direction of the coherent motion." (Section 4.1 Toy Example)
- "we adapted this task into a two-image input scenario." (Section 4.1 Toy Example)
- "The CNN would receive an original image and its shifted counterpart — the shift can be to the left, right, up, or down." (Section 4.1 Toy Example)
- "on this 4-class task" (Section 4.1 Toy Example)
- Inference: Inferred 3D (x, y, t) and fixed input dynamics from "two-image input scenario" and "original image and its shifted counterpart"; inferred 0D label output from "identify the direction of the coherent motion" and "4-class task"; inferred Dynamic attention and Constructed state from the introspection network description "dynamically adjusting the computational graph" and "analyzing intermediate activations." (Sections 4.1, 3.1)

### Task: Language modeling
- "We evaluated performance on language modeling tasks across multiple datasets including WikiText-2 and WikiText-103 datasets (Merity et al., 2016)." (Section 4.4 Experiments on Language Modeling Tasks)
- "WikiText-2 contains 2 million tokens, while WikiText-103 consists of 103 million tokens" (Language Modeling)
- "Max Sequence Length            | 512" (Table 1)
- Inference: Inferred 1D token sequences, capped dynamics, and token outputs from "tokens" in WikiText and the "Max Sequence Length" setting; inferred Dynamic attention and Constructed state because the introspection network "assesses the complexity of the input and the current activation states" and is "responsible for analyzing intermediate activations." (Sections Language Modeling, 3.1)

### Task: Question answering
- "SQuAD v2.0 was employed for question-answering, where Exact Match (EM) and F1 scores were reported." (Language Modeling)
- "the approach achieves 95.8%/88.7% F1 scores on the SQuAD v1.1/v2.0 datasets" (Abstract)
- Inference: Inferred 1D text inputs/outputs and capped dynamics from the question-answering setup and the transformer "Max Sequence Length"; inferred Dynamic attention and Constructed state because the introspection network "dynamically adjusting the computational graph" and is "responsible for analyzing intermediate activations." (Sections Language Modeling, 3.1)

### Task: Summarization
- "CNN/DailyMail (Nallapati et al., 2016) for summarization" (Section F.2 Early Exit Experiments)
- "tested on WikiText-103 for language modeling, CNN/DailyMail (Nallapati et al., 2016) for summarization, and SQuAD v2.0 (Rajpurkar et al., 2018) for question answering." (Section F.2 Early Exit Experiments)
- Inference: Inferred 1D document/summary text inputs and capped dynamics from the summarization setup and the transformer "Max Sequence Length"; inferred Dynamic attention and Constructed state because the introspection network "assesses the complexity of the input and the current activation states" and is "responsible for analyzing intermediate activations." (Sections F.2, 3.1)
