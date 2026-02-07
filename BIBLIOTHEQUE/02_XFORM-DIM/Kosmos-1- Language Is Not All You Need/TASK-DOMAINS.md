# Language Is Not All You Need: Aligning Perception with Language Models (Not specified in the paper)
Source: Kosmos-1- Language Is Not All You Need.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Commonsense reasoning (StoryCloze) | text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text completion (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Commonsense NLI (HellaSwag) | text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text completion (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Word ambiguity (Winograd) | text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text answer (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Word ambiguity (Winogrande) | text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text answer (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Physical commonsense (PIQA) | text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text answer (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Question answering (BoolQ) | text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text answer (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Textual entailment (CB) | text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text answer (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Causal reasoning (COPA) | text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text answer (inferred) | 1D (t) (inferred) | Capped (inferred) |
| OCR-free sentiment classification (Rendered SST-2) | rendered text images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | sentiment label (positive/negative) | 1D (t) (inferred) | Capped (inferred) |
| OCR-free meme classification (HatefulMemes) | meme images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Yes/No answer | 1D (t) (inferred) | Capped (inferred) |
| Object size reasoning (RelativeSize) | text prompts (object pairs) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Yes/No answer | 1D (t) (inferred) | Capped (inferred) |
| Object color reasoning (MemoryColor) | text prompts (object) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | color label | 1D (t) (inferred) | Capped (inferred) |
| Object color reasoning (ColorTerms) | text prompts (object) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | color label | 1D (t) (inferred) | Capped (inferred) |
| Raven's Progressive Matrices (IQ Test) | matrix images + textual instruction | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Yes/No answer | 1D (t) (inferred) | Capped (inferred) |
| Image captioning (COCO Caption) | image | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | natural language description | 1D (t) (inferred) | Capped (inferred) |
| Image captioning (Flicker30k) | image | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | natural language description | 1D (t) (inferred) | Capped (inferred) |
| Visual question answering (VQAv2) | image + question text | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text answer | 1D (t) (inferred) | Capped (inferred) |
| Visual question answering (VizWiz) | image + question text | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text answer | 1D (t) (inferred) | Capped (inferred) |
| Web page question answering (WebSRC) | web page image + extracted text + question | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | answer from web page | 1D (t) (inferred) | Capped (inferred) |
| Zero-shot image classification (ImageNet) | image | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | category name | 1D (t) (inferred) | Capped (inferred) |
| Zero-shot image classification with descriptions (CUB) | image + category descriptions (text) | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | specific category name | 1D (t) (inferred) | Capped (inferred) |

## Summary
KOSMOS-1 is evaluated on text-only language reasoning tasks, OCR-free classification from rendered text/memes, nonverbal reasoning, perception-language tasks (image captioning, VQA, web page QA), and vision classification with and without descriptions. Inputs span text sequences and images, as well as mixed image+text prompts, while outputs are textual answers or descriptions. Dimension and dynamics assignments are inferred from the model's fixed 224x224 image preprocessing and 2,048-token max length, yielding 1D (t) and 2D (x, y) inputs with mostly capped interfaces. Attention is Static and state is Direct, inferred from the causal Transformer decoder.

## Evidence
### Task: Commonsense reasoning (StoryCloze)
- "StoryCloze [MRL+17]                  | Commonsense reasoning" (Table 1)
- "Text inputs are directly fed into the models as in vanilla language models." (Section 4.8)
- Inference: In/Out dimensions (1D (t)) and capped dynamics are inferred from the 2,048 max length (Table 17); text output and Static/Direct dynamics are inferred from the auto-regressive decoder (Section 2.2).

### Task: Commonsense NLI (HellaSwag)
- "HellaSwag [ZHB+19]                   | Commonsense NLI" (Table 1)
- "Text inputs are directly fed into the models as in vanilla language models." (Section 4.8)
- Inference: In/Out dimensions (1D (t)) and capped dynamics are inferred from the 2,048 max length (Table 17); text output and Static/Direct dynamics are inferred from the auto-regressive decoder (Section 2.2).

### Task: Word ambiguity (Winograd)
- "Winograd [LDM12a]                    | Word ambiguity" (Table 1)
- "Text inputs are directly fed into the models as in vanilla language models." (Section 4.8)
- Inference: In/Out dimensions (1D (t)) and capped dynamics are inferred from the 2,048 max length (Table 17); text output and Static/Direct dynamics are inferred from the auto-regressive decoder (Section 2.2).

### Task: Word ambiguity (Winogrande)
- "Winogrande [SBBC20]                  | Word ambiguity" (Table 1)
- "Text inputs are directly fed into the models as in vanilla language models." (Section 4.8)
- Inference: In/Out dimensions (1D (t)) and capped dynamics are inferred from the 2,048 max length (Table 17); text output and Static/Direct dynamics are inferred from the auto-regressive decoder (Section 2.2).

### Task: Physical commonsense (PIQA)
- "PIQA [BZB <sup>+</sup> 20]           | Physical commonsense" (Table 1)
- "Text inputs are directly fed into the models as in vanilla language models." (Section 4.8)
- Inference: In/Out dimensions (1D (t)) and capped dynamics are inferred from the 2,048 max length (Table 17); text output and Static/Direct dynamics are inferred from the auto-regressive decoder (Section 2.2).

### Task: Question answering (BoolQ)
- "BoolQ [CLC+19]                       | Question answering" (Table 1)
- "Text inputs are directly fed into the models as in vanilla language models." (Section 4.8)
- Inference: In/Out dimensions (1D (t)) and capped dynamics are inferred from the 2,048 max length (Table 17); text output and Static/Direct dynamics are inferred from the auto-regressive decoder (Section 2.2).

### Task: Textual entailment (CB)
- "CB [dMST19]                          | Textual entailment" (Table 1)
- "Text inputs are directly fed into the models as in vanilla language models." (Section 4.8)
- Inference: In/Out dimensions (1D (t)) and capped dynamics are inferred from the 2,048 max length (Table 17); text output and Static/Direct dynamics are inferred from the auto-regressive decoder (Section 2.2).

### Task: Causal reasoning (COPA)
- "COPA [RBG11]                         | Causal reasoning" (Table 1)
- "Text inputs are directly fed into the models as in vanilla language models." (Section 4.8)
- Inference: In/Out dimensions (1D (t)) and capped dynamics are inferred from the 2,048 max length (Table 17); text output and Static/Direct dynamics are inferred from the auto-regressive decoder (Section 2.2).

### Task: OCR-free sentiment classification (Rendered SST-2)
- "Rendered SST-2 [RKH <sup>+</sup> 21] | OCR-free sentiment classification" (Table 1)
- "sentences from the Stanford Sentiment Treebank [SPW+13] dataset are rendered as images." (Section 4.3)
- Inference: In Dimension 2D (x, y) and Fixed dynamics are inferred from 224x224 image preprocessing (Section 3.2); Out Dimension 1D (t), capped dynamics, and Static/Direct are inferred from the auto-regressive decoder and max length 2,048 (Section 2.2; Table 17).

### Task: OCR-free meme classification (HatefulMemes)
- "HatefulMemes [KFM <sup>+</sup> 20]   | OCR-free meme classification" (Table 1)
- "Question: does this picture contain real hate speech? Answer: {answer}" (Section 4.3.1)
- Inference: In Dimension 2D (x, y) and Fixed dynamics are inferred from 224x224 image preprocessing (Section 3.2); Out Dimension 1D (t), capped dynamics, and Static/Direct are inferred from the auto-regressive decoder and max length 2,048 (Section 2.2; Table 17).

### Task: Object size reasoning (RelativeSize)
- "RelativeSize [BHCF16]                | Commonsense reasoning (object size)" (Table 1)
- "predict the size relation between two objects in a binary question-answering format with "Yes"/"No" answers." (Section 4.9.2)
- Inference: In/Out dimensions (1D (t)) and capped dynamics are inferred from the 2,048 max length (Table 17); Static/Direct dynamics are inferred from the auto-regressive decoder (Section 2.2).

### Task: Object color reasoning (MemoryColor)
- "MemoryColor [NHJ21]                  | Commonsense reasoning (object color)" (Table 1)
- "MemoryColor and ColorTerms require the model to predict the color of objects from a set of 11 color labels in a multiple-choice format." (Section 4.9.2)
- Inference: In/Out dimensions (1D (t)) and capped dynamics are inferred from the 2,048 max length (Table 17); Static/Direct dynamics are inferred from the auto-regressive decoder (Section 2.2).

### Task: Object color reasoning (ColorTerms)
- "ColorTerms [BBBT12]                  | Commonsense reasoning (object color)" (Table 1)
- "MemoryColor and ColorTerms require the model to predict the color of objects from a set of 11 color labels in a multiple-choice format." (Section 4.9.2)
- Inference: In/Out dimensions (1D (t)) and capped dynamics are inferred from the 2,048 max length (Table 17); Static/Direct dynamics are inferred from the auto-regressive decoder (Section 2.2).

### Task: Raven's Progressive Matrices (IQ Test)
- "The input prompt consists of the flattened image matrix and verbal instruction." (Figure 4)
- "The final prediction is the candidate that motivates the model to yield the highest probability of "Yes"." (Figure 4)
- Inference: In Dimension includes 2D (x, y) and 1D (t) and capped dynamics are inferred from 224x224 image preprocessing and max length 2,048 (Section 3.2; Table 17); Out Dimension 1D (t) and Static/Direct are inferred from the auto-regressive decoder (Section 2.2).

### Task: Image captioning (COCO Caption)
- "COCO Caption [LMB <sup>+</sup> 14]   | Image captioning" (Table 1)
- "Image captioning involves generating a natural language description of an image." (Section 4.1)
- Inference: In Dimension 2D (x, y) and Fixed dynamics are inferred from 224x224 image preprocessing (Section 3.2); Out Dimension 1D (t), capped dynamics, and Static/Direct are inferred from the auto-regressive decoder and max length 2,048 (Section 2.2; Table 17).

### Task: Image captioning (Flicker30k)
- "Flicker30k [YLHH14]                  | Image captioning" (Table 1)
- "Image captioning involves generating a natural language description of an image." (Section 4.1)
- Inference: In Dimension 2D (x, y) and Fixed dynamics are inferred from 224x224 image preprocessing (Section 3.2); Out Dimension 1D (t), capped dynamics, and Static/Direct are inferred from the auto-regressive decoder and max length 2,048 (Section 2.2; Table 17).

### Task: Visual question answering (VQAv2)
- "VQAv2 [GKSS <sup>+</sup> 17]         | Visual question answering" (Table 1)
- "visual question answering aims to answer a natural language question with respect to an image." (Section 4.1)
- Inference: In Dimension includes 2D (x, y) and 1D (t) and capped dynamics are inferred from 224x224 image preprocessing and max length 2,048 (Section 3.2; Table 17); Out Dimension 1D (t) and Static/Direct are inferred from the auto-regressive decoder (Section 2.2).

### Task: Visual question answering (VizWiz)
- "VizWiz [GLS <sup>+</sup> 18]         | Visual question answering" (Table 1)
- "visual question answering aims to answer a natural language question with respect to an image." (Section 4.1)
- Inference: In Dimension includes 2D (x, y) and 1D (t) and capped dynamics are inferred from 224x224 image preprocessing and max length 2,048 (Section 3.2; Table 17); Out Dimension 1D (t) and Static/Direct are inferred from the auto-regressive decoder (Section 2.2).

### Task: Web page question answering (WebSRC)
- "WebSRC [CZC <sup>+</sup> 21]         | Web page question answering" (Table 1)
- "Web page question answering aims at finding answers to questions from web pages." (Section 4.4)
- Inference: In Dimension includes 2D (x, y) and 1D (t) and capped dynamics are inferred from 224x224 image preprocessing and max length 2,048 (Section 3.2; Table 17); Out Dimension 1D (t) and Static/Direct are inferred from the auto-regressive decoder (Section 2.2).

### Task: Zero-shot image classification (ImageNet)
- "ImageNet [DDS+09]                    | Zero-shot image classification" (Table 1)
- "Image classification comprehends an entire image as a whole and aims to assign a label to the image." (Section 4.6)
- Inference: In Dimension 2D (x, y) and Fixed dynamics are inferred from 224x224 image preprocessing (Section 3.2); Out Dimension 1D (t), capped dynamics, and Static/Direct are inferred from the auto-regressive decoder and max length 2,048 (Section 2.2; Table 17).

### Task: Zero-shot image classification with descriptions (CUB)
- "CUB [WBW <sup>+</sup> 11]            | Zero-shot image classification with descriptions" (Table 1)
- "Our goal is to classify images given the categories' descriptions." (Section 4.7.1)
- Inference: In Dimension includes 2D (x, y) and 1D (t) and capped dynamics are inferred from 224x224 image preprocessing and max length 2,048 (Section 3.2; Table 17); Out Dimension 1D (t) and Static/Direct are inferred from the auto-regressive decoder (Section 2.2).
