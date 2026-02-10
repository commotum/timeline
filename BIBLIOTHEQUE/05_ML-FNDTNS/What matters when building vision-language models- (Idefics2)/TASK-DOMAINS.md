# What matters when building vision-language models? (2024)
Source: What matters when building vision-language models- (Idefics2).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Visual question answering | images; question tokens | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | answer tokens | 1D (t) (inferred) | Capped (inferred) |
| Counting objects in images | images; question tokens | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | count tokens | 1D (t) (inferred) | Capped (inferred) |
| Image captioning | images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | caption tokens | 1D (t) (inferred) | Capped (inferred) |
| Text transcription (OCR) | images/documents | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | transcribed text tokens | 1D (t) (inferred) | Capped (inferred) |
| Document understanding | document images/PDF pages; question tokens | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | answer tokens | 1D (t) (inferred) | Capped (inferred) |
| Chart/figure understanding | chart/figure images; question tokens | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | explanation/answer tokens | 1D (t) (inferred) | Capped (inferred) |
| Table understanding | table images; question tokens | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | answer tokens | 1D (t) (inferred) | Capped (inferred) |
| Visual reasoning | images; question tokens | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | reasoning answer tokens | 1D (t) (inferred) | Capped (inferred) |
| Geometry reasoning | geometry diagrams/images; question tokens | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | geometry answer tokens | 1D (t) (inferred) | Capped (inferred) |
| Spotting differences between 2 images | two images; question tokens | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | difference description tokens | 1D (t) (inferred) | Capped (inferred) |
| Screenshot-to-code generation | webpage screenshot image; instruction/question tokens | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | code tokens | 1D (t) (inferred) | Capped (inferred) |
| Complex instruction following (text-only) | instruction tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | response tokens | 1D (t) (inferred) | Capped (inferred) |
| Mathematical problem solving | text and/or image problem statements; question tokens | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | solution tokens | 1D (t) (inferred) | Capped (inferred) |
| Arithmetic calculation | text and/or table/image context; question tokens | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | numeric answer tokens | 1D (t) (inferred) | Capped (inferred) |
| Chat dialogue response generation | images and/or text; multi-turn conversation tokens | 1D (t); 2D (x, y) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | dialogue response tokens | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper describes Idefics2 as a multimodal text-generating VLM covering broad vision-language tasks plus text-only instruction data. Supported tasks span visual QA, OCR/document/chart/table tasks, reasoning (including geometry), counting, captioning, difference spotting, screenshot-to-code, math/arithmetic, and chat dialogue generation. Inputs cover 2D visual objects and 1D token sequences, while outputs are text tokens (1D). Based on explicit maximum sequence lengths and resolution limits, most interfaces are Capped; chat is Open due ongoing multi-turn interaction, with Static attention and Direct state inferred from the described autoregressive architecture.

## Evidence
### Task: Visual question answering
- "To do so, we create and release The Cauldron<sup>8</sup>, a massive collection of 50 vision-language datasets, covering a wide range of tasks: general visual question answering, counting, captioning, text transcription, document understanding, chart/figure understanding, table understanding, visual reasoning, geometry, spotting differences between 2 images or converting a screenshot to a functional code." (Section 4.2 Instruction fine-tuning)
- "In this section, we compare recurrent design choices in the vision-language model literature and highlight findings. Unless specified otherwise, we run the ablations for 6'000 steps and report the average score of the 4-shot performance on 4 downstream benchmarks measuring different capabilities: VQAv2 (Goyal et al., 2017) for general visual question answering, TextVQA (Singh et al., 2019) for OCR abilities, OKVQA (Marino et al., 2019) for external knowledge, and COCO (Lin et al., 2014) for captioning." (Section 3 Exploring the design space of vision-language models)
- Inference: Input/output dimensions and dynamics labels are inferred from "take images and texts as inputs and output texts," plus explicit "maximum sequence length" constraints and autoregressive token generation. (Section 1 Introduction; Section 4.1 Multi-stage pre-training; Figure 2)

### Task: Counting objects in images
- "To do so, we create and release The Cauldron<sup>8</sup>, a massive collection of 50 vision-language datasets, covering a wide range of tasks: general visual question answering, counting, captioning, text transcription, document understanding, chart/figure understanding, table understanding, visual reasoning, geometry, spotting differences between 2 images or converting a screenshot to a functional code." (Section 4.2 Instruction fine-tuning)
- "Vision-language models (VLMs) that take images and texts as inputs and output texts, are useful for many tasks, like retrieving information in a scanned PDF (Hu et al., 2024), explaining charts or diagrams (Carbune et al., 2024), transcribing the text in an image (Blecher et al., 2023), counting objects in a picture (Goyal et al., 2017) or turning screenshots of webpages into code (Laurençon et al., 2024)." (Section 1 Introduction)
- Inference: Counting is implemented in the shared Q/A text-output setup, so dimensions and dynamics are mapped to 2D visual input + 1D text and capped sequence processing. (Section 4.2 Instruction fine-tuning; Section 4.1 Multi-stage pre-training)

### Task: Image captioning
- "To do so, we create and release The Cauldron<sup>8</sup>, a massive collection of 50 vision-language datasets, covering a wide range of tasks: general visual question answering, counting, captioning, text transcription, document understanding, chart/figure understanding, table understanding, visual reasoning, geometry, spotting differences between 2 images or converting a screenshot to a functional code." (Section 4.2 Instruction fine-tuning)
- "In this section, we compare recurrent design choices in the vision-language model literature and highlight findings. Unless specified otherwise, we run the ablations for 6'000 steps and report the average score of the 4-shot performance on 4 downstream benchmarks measuring different capabilities: VQAv2 (Goyal et al., 2017) for general visual question answering, TextVQA (Singh et al., 2019) for OCR abilities, OKVQA (Marino et al., 2019) for external knowledge, and COCO (Lin et al., 2014) for captioning." (Section 3 Exploring the design space of vision-language models)
- Inference: Captioning output is classified as token generation (1D) because the model is described as outputting text tokens. (Section 1 Introduction; Figure 2)

### Task: Text transcription (OCR)
- "To do so, we create and release The Cauldron<sup>8</sup>, a massive collection of 50 vision-language datasets, covering a wide range of tasks: general visual question answering, counting, captioning, text transcription, document understanding, chart/figure understanding, table understanding, visual reasoning, geometry, spotting differences between 2 images or converting a screenshot to a functional code." (Section 4.2 Instruction fine-tuning)
- "**PDF documents** Sun et al. (2023) shows that a large proportion of mistakes of state-of-the art VLMs stem from their failure to accurately extract text in images or documents. In order to obtain strong OCR and document understanding abilities, we train Idefics2 on different sources of PDF documents: 19 million industry documents from OCR-IDL (Biten et al., 2022) and 18 million pages from PDFA<sup>6</sup>." (Section 4.1 Multi-stage pre-training)
- Inference: OCR dimensions/dynamics are inferred as 2D visual input to 1D text output under capped resolution/sequence limits. (Section 4.1 Multi-stage pre-training)

### Task: Document understanding
- "To do so, we create and release The Cauldron<sup>8</sup>, a massive collection of 50 vision-language datasets, covering a wide range of tasks: general visual question answering, counting, captioning, text transcription, document understanding, chart/figure understanding, table understanding, visual reasoning, geometry, spotting differences between 2 images or converting a screenshot to a functional code." (Section 4.2 Instruction fine-tuning)
- "**PDF documents** Sun et al. (2023) shows that a large proportion of mistakes of state-of-the art VLMs stem from their failure to accurately extract text in images or documents. In order to obtain strong OCR and document understanding abilities, we train Idefics2 on different sources of PDF documents: 19 million industry documents from OCR-IDL (Biten et al., 2022) and 18 million pages from PDFA<sup>6</sup>." (Section 4.1 Multi-stage pre-training)
- Inference: Document understanding is represented as document-image + text-question to text-answer in the shared Q/A format; dimension/dynamics labels are inferred from that setup and max length constraints. (Section 4.2 Instruction fine-tuning; Section 4.1 Multi-stage pre-training)

### Task: Chart/figure understanding
- "To do so, we create and release The Cauldron<sup>8</sup>, a massive collection of 50 vision-language datasets, covering a wide range of tasks: general visual question answering, counting, captioning, text transcription, document understanding, chart/figure understanding, table understanding, visual reasoning, geometry, spotting differences between 2 images or converting a screenshot to a functional code." (Section 4.2 Instruction fine-tuning)
- "Vision-language models (VLMs) that take images and texts as inputs and output texts, are useful for many tasks, like retrieving information in a scanned PDF (Hu et al., 2024), explaining charts or diagrams (Carbune et al., 2024), transcribing the text in an image (Blecher et al., 2023), counting objects in a picture (Goyal et al., 2017) or turning screenshots of webpages into code (Laurençon et al., 2024)." (Section 1 Introduction)
- Inference: Chart/figure understanding is mapped to 2D visual + 1D textual input and 1D text output in capped sequence settings. (Section 1 Introduction; Section 4.1 Multi-stage pre-training)

### Task: Table understanding
- "To do so, we create and release The Cauldron<sup>8</sup>, a massive collection of 50 vision-language datasets, covering a wide range of tasks: general visual question answering, counting, captioning, text transcription, document understanding, chart/figure understanding, table understanding, visual reasoning, geometry, spotting differences between 2 images or converting a screenshot to a functional code." (Section 4.2 Instruction fine-tuning)
- "Figure 1: Idefics2-chatty analyzes the table to compute and answer the query." (Figure 1)
- Inference: Table tasks are treated as 2D structured visual/table input plus text prompts in Q/A form, producing text answers. (Section 1 Introduction; Section 4.2 Instruction fine-tuning)

### Task: Visual reasoning
- "To do so, we create and release The Cauldron<sup>8</sup>, a massive collection of 50 vision-language datasets, covering a wide range of tasks: general visual question answering, counting, captioning, text transcription, document understanding, chart/figure understanding, table understanding, visual reasoning, geometry, spotting differences between 2 images or converting a screenshot to a functional code." (Section 4.2 Instruction fine-tuning)
- "(Singh et al., 2019) for text reading on natural images, and MMBench Liu et al. (2023) for various perception and reasoning tasks." (Section 4.2 Instruction fine-tuning)
- Inference: Reasoning tasks are implemented through shared question/answer token generation over image+text inputs, with capped interface limits. (Section 4.2 Instruction fine-tuning; Section 4.1 Multi-stage pre-training)

### Task: Geometry reasoning
- "To do so, we create and release The Cauldron<sup>8</sup>, a massive collection of 50 vision-language datasets, covering a wide range of tasks: general visual question answering, counting, captioning, text transcription, document understanding, chart/figure understanding, table understanding, visual reasoning, geometry, spotting differences between 2 images or converting a screenshot to a functional code." (Section 4.2 Instruction fine-tuning)
- "| Reasoning, logic, maths GeomVerse (Kazemi et al., 2024) CLEVR-Math (Lindström, 2022) CLEVR (Johnson et al., 2017) IconQA (Lu et al., 2021) RAVEN (Zhang et al., 2019) Inter-GPs (Lu et al., 2021)" (Section A.2.1 Statistics of The Cauldron)
- Inference: Geometry reasoning is mapped to visual+text QA with token outputs because all datasets are converted into shared Q/A format. (Section 4.2 Instruction fine-tuning)

### Task: Spotting differences between 2 images
- "To do so, we create and release The Cauldron<sup>8</sup>, a massive collection of 50 vision-language datasets, covering a wide range of tasks: general visual question answering, counting, captioning, text transcription, document understanding, chart/figure understanding, table understanding, visual reasoning, geometry, spotting differences between 2 images or converting a screenshot to a functional code." (Section 4.2 Instruction fine-tuning)
- "| Textbook/academic questions AI2D (Kembhavi et al., 2016) TQA (Kembhavi et al., 2017) ScienceQA (Lu et al., 2022)  Differences between 2 images" (Section A.2.1 Statistics of The Cauldron)
- Inference: This task uses paired 2D image inputs plus textual questioning in the shared Q/A format and outputs text descriptions. (Section 4.2 Instruction fine-tuning)

### Task: Screenshot-to-code generation
- "To do so, we create and release The Cauldron<sup>8</sup>, a massive collection of 50 vision-language datasets, covering a wide range of tasks: general visual question answering, counting, captioning, text transcription, document understanding, chart/figure understanding, table understanding, visual reasoning, geometry, spotting differences between 2 images or converting a screenshot to a functional code." (Section 4.2 Instruction fine-tuning)
- "Vision-language models (VLMs) that take images and texts as inputs and output texts, are useful for many tasks, like retrieving information in a scanned PDF (Hu et al., 2024), explaining charts or diagrams (Carbune et al., 2024), transcribing the text in an image (Blecher et al., 2023), counting objects in a picture (Goyal et al., 2017) or turning screenshots of webpages into code (Laurençon et al., 2024)." (Section 1 Introduction)
- Inference: Screenshot-to-code is classified as 2D screenshot + 1D instruction tokens to 1D code-token generation under capped sequence limits. (Section 1 Introduction; Section 4.1 Multi-stage pre-training)

### Task: Complex instruction following (text-only)
- "In addition to these vision-language datasets and following insights from (McKinzie et al., 2024), we add text-only instruction datasets to the mixture. The datasets aim at teaching the model to follow complex instructions, solve mathematical problems, or do arithmetic calculations." (Section 4.2 Instruction fine-tuning)
- "| Text-only general instructions, math pr<br>OpenHermes-2.5 (Teknium, 2023)<br>LIMA (Zhou et al., 2023)<br>Dolly (Conover et al., 2023)<br>MetaMathQA (Yu et al., 2024)<br>MathInstruct (Yue et al., 2024)" (Section A.2.1 Statistics of The Cauldron)
- Inference: Dimension is inferred as 1D token sequences for text-only instructions; dynamics are inferred as capped from explicit sequence-length limits used in training. (Section 4.1 Multi-stage pre-training)

### Task: Mathematical problem solving
- "In addition to these vision-language datasets and following insights from (McKinzie et al., 2024), we add text-only instruction datasets to the mixture. The datasets aim at teaching the model to follow complex instructions, solve mathematical problems, or do arithmetic calculations." (Section 4.2 Instruction fine-tuning)
- "We evaluate Idefics2 on commonly adopted benchmarks: MMMU (Yue et al., 2024) for multidiscipline college-level problems, MathVista (Lu et al., 2024) for mathematical reasoning," (Section 4.2 Instruction fine-tuning)
- Inference: This is modeled as token generation over text-only and visual+text math contexts; therefore the input dimension is marked 1D and 2D (inferred). (Section 4.2 Instruction fine-tuning)

### Task: Arithmetic calculation
- "In addition to these vision-language datasets and following insights from (McKinzie et al., 2024), we add text-only instruction datasets to the mixture. The datasets aim at teaching the model to follow complex instructions, solve mathematical problems, or do arithmetic calculations." (Section 4.2 Instruction fine-tuning)
- "Figure 1: Idefics2-chatty analyzes the table to compute and answer the query." (Figure 1)
- Inference: Arithmetic is inferred as answer-token generation from text and/or visual-tabular contexts in the same Q/A generative interface. (Section 1 Introduction; Section 4.2 Instruction fine-tuning)

### Task: Chat dialogue response generation
- "When there are multiple question/answer pairs per image, we concatenate the pairs into a multi-turn conversation." (Section 4.2 Instruction fine-tuning)
- "The evaluation benchmarks expect very short answers, but humans prefer long generations when interacting with a model." (Section 4.3 Optimizing for chat scenarios)
- "As such, after instruction fine-tuning, we further train Idefics2 on dialogue data." (Section 4.3 Optimizing for chat scenarios)
- Inference: Input/output dynamics are marked Open (inferred) because the paper explicitly describes ongoing multi-turn interaction and chat optimization rather than a single fixed-length exchange. (Section 4.2 Instruction fine-tuning; Section 4.3 Optimizing for chat scenarios)
