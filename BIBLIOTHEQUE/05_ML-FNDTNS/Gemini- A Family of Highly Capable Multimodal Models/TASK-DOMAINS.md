# Gemini: A Family of Highly Capable Multimodal Models (Not specified in the paper)
Source: Gemini- A Family of Highly Capable Multimodal Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Text completion / generation | Text prompt / tokens | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Generated text | 1D (t) (inferred) | Not specified in the paper. |
| Text summarization | Text documents | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Summary text | 1D (t) (inferred) | Not specified in the paper. |
| Text question answering / reading comprehension | Question + text passages / retrieved context | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Answer text | 1D (t) (inferred) | Not specified in the paper. |
| Machine translation | Source-language text | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Translated text | 1D (t) (inferred) | Not specified in the paper. |
| Math/science problem solving | Math/science problems (text) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Solutions / answers (text) | 1D (t) (inferred) | Not specified in the paper. |
| Code generation / completion | Natural-language function description / prompt | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Code (e.g., Python) | 1D (t) (inferred) | Not specified in the paper. |
| Image captioning / description | Images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Image descriptions (text) | 1D (t) (inferred) | Not specified in the paper. |
| Visual question answering (natural images) | Image + text question | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Short text answer | 1D (t) (inferred) | Not specified in the paper. |
| Visual transcription / OCR | Images/documents with text | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Transcribed text | 1D (t) (inferred) | Not specified in the paper. |
| Chart/infographic question answering | Charts/infographics + question | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Text answer | 1D (t) (inferred) | Not specified in the paper. |
| Multimodal reasoning on images/diagrams | Images/diagrams + text question | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Text answer | 1D (t) (inferred) | Not specified in the paper. |
| Image generation | Interleaved image + text prompt | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Generated images interleaved with text | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. |
| Video captioning | Video (sequence of frames) | 3D (x, y, t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Text caption | 1D (t) (inferred) | Not specified in the paper. |
| Video question answering | Video + text question | 3D (x, y, t); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Text answer | 1D (t) (inferred) | Not specified in the paper. |
| Speech recognition (ASR) | Audio | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Transcribed text | 1D (t) (inferred) | Not specified in the paper. |
| Speech translation | Audio | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Translated text | 1D (t) (inferred) | Not specified in the paper. |
| Multimodal audio-visual dialog / instruction following | Interleaved audio + images + text prompts | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Text responses / instructions | 1D (t) (inferred) | Not specified in the paper. |

## Summary
Gemini is evaluated on text tasks such as completion, summarization, question answering/reading comprehension, translation, math/science problem solving, and code generation, plus multimodal tasks over images, audio, and video including captioning, question answering, OCR/transcription, chart/diagram reasoning, image generation, video captioning/QA, and speech recognition/translation. It also demonstrates multimodal dialog over interleaved audio and images. Based on the modalities described, inputs span 1D text/audio, 2D images, and 3D video with outputs primarily 1D text plus 2D image outputs for generation; dynamics/attention/state are not explicitly specified in the paper.

## Evidence
### Task: Text completion / generation
- "These models excel in on-device tasks, such as summarization, reading comprehension, text completion tasks" (Section 1 Introduction)
- Inference: In/Out Dimension marked as 1D (t) because the task is explicitly text completion. (inferred)

### Task: Text summarization
- "covering long-form summarization, retrieval and question answering tasks" (Section 5.1.2)
- Inference: In/Out Dimension marked as 1D (t) because summarization is a text task. (inferred)

### Task: Text question answering / reading comprehension
- "covering open/closed-book retrieval and question answering tasks" (Section 5.1.2)
- "benchmarks covering reasoning, reading comprehension, STEM, and coding." (Section 5.1.1)
- Inference: In/Out Dimension marked as 1D (t) because QA/reading comprehension are text tasks. (inferred)

### Task: Machine translation
- "These tasks include machine translation benchmarks (WMT 23 for high-medium-low resource translation; Flores, NTREX for low and very low resource languages)" (Section 5.1.4)
- Inference: In/Out Dimension marked as 1D (t) because translation is text-to-text. (inferred)

### Task: Math/science problem solving
- "including tasks for mathematical problem solving, theorem proving, and scientific exams" (Section 5.1.2)
- Inference: In/Out Dimension marked as 1D (t) because these problems are presented and answered in text. (inferred)

### Task: Code generation / completion
- "HumanEval, a standard code-completion benchmark (Chen et al., 2021) mapping function descriptions to Python implementations" (Section 5.1.1)
- Inference: In/Out Dimension marked as 1D (t) because both prompts and code are token sequences. (inferred)

### Task: Image captioning / description
- "high-level object recognition using captioning or question-answering tasks such as VQAv2" (Section 5.2.1)
- "generating image descriptions for a wide range of languages" (Section 5.2.1)
- Inference: In Dimension marked as 2D (x, y) and Out Dimension as 1D (t) based on image inputs and text descriptions. (inferred)

### Task: Visual question answering (natural images)
- "answering questions on natural images and scanned documents" (Section 5.2.1)
- Inference: In Dimension marked as 2D (x, y); 1D (t) because the task combines images with text questions; output is text. (inferred)

### Task: Visual transcription / OCR
- "fine-grained transcription using tasks such as TextVQA and DocVQA requiring the model to recognize low-level details" (Section 5.2.1)
- Inference: In Dimension marked as 2D (x, y) and Out Dimension as 1D (t) because the task transcribes text from images. (inferred)

### Task: Chart/infographic question answering
- "chart understanding requiring spatial understanding of input layout using ChartQA and InfographicVQA tasks" (Section 5.2.1)
- Inference: In Dimension marked as 2D (x, y); 1D (t) because charts are images paired with text questions; output is text. (inferred)

### Task: Multimodal reasoning on images/diagrams
- "multimodal reasoning using tasks such as Ai2D, MathVista and MMMU" (Section 5.2.1)
- Inference: In Dimension marked as 2D (x, y); 1D (t) because the tasks involve images with text prompts; output is text. (inferred)

### Task: Image generation
- "Gemini models are able to output images natively" (Section 5.2.3)
- "generate images with prompts using interleaved sequences of image and text" (Section 5.2.3)
- "Gemini models can output multiple images interleaved with text given a prompt composed of image and text." (Figure 6 caption)
- Inference: In/Out Dimensions marked as 2D (x, y) and 1D (t) because inputs and outputs are interleaved images and text. (inferred)

### Task: Video captioning
- "video captioning tasks" (Section 5.2.2)
- Inference: In Dimension marked as 3D (x, y, t) because video is a temporal sequence of frames; output is text. (inferred)

### Task: Video question answering
- "video question answering tasks" (Section 5.2.2)
- Inference: In Dimension marked as 3D (x, y, t); 1D (t) because the task combines video with text questions; output is text. (inferred)

### Task: Speech recognition (ASR)
- "These benchmarks include automatic speech recognition (ASR) tasks such as FLEURS" (Section 5.2.4)
- Inference: In/Out Dimension marked as 1D (t) because speech and transcripts are temporal/text sequences. (inferred)

### Task: Speech translation
- "as well as the speech translation task CoVoST 2, translating different languages into English" (Section 5.2.4)
- Inference: In/Out Dimension marked as 1D (t) because audio inputs are temporal and outputs are text. (inferred)

### Task: Multimodal audio-visual dialog / instruction following
- "We demonstrate the ability to process a sequence of audio and images natively." (Section 5.2.5)
- "The user prompts the model for instructions to make an omelet and to inspect whether it is fully cooked." (Table 13)
- Inference: In Dimension marked as 2D (x, y); 1D (t) because the inputs interleave images, audio, and text; output is text. (inferred)
