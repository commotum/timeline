## 1. Basic Metadata

- Title: Language Is Not All You Need: Aligning Perception with Language Models
- Authors: Shaohan Huang*, Li Dong*, Wenhui Wang*, Yaru Hao*, Saksham Singhal*, Shuming Ma*, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, Furu Wei, Microsoft
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

The paper introduces KOSMOS-1, a multimodal large language model intended to align perception with language models so it can handle multimodal inputs in zero-/few-shot settings.

## 3. Tasks Evaluated

- StoryCloze - Task type: Classification; Reasoning / relational. Dataset(s): StoryCloze [MRL+17]. Domain: text. Evidence: "StoryCloze [MRL+17]" and "Commonsense reasoning" (Table 1).
- HellaSwag - Task type: Classification; Reasoning / relational. Dataset(s): HellaSwag [ZHB+19]. Domain: text. Evidence: "HellaSwag [ZHB+19]" and "Commonsense NLI" (Table 1).
- Winograd - Task type: Classification; Reasoning / relational. Dataset(s): Winograd [LDM12a]. Domain: text. Evidence: "Winograd [LDM12a]" and "Word ambiguity" (Table 1).
- Winogrande - Task type: Classification; Reasoning / relational. Dataset(s): Winogrande [SBBC20]. Domain: text. Evidence: "Winogrande [SBBC20]" and "Word ambiguity" (Table 1).
- PIQA - Task type: Classification; Reasoning / relational. Dataset(s): PIQA [BZB <sup>+</sup> 20]. Domain: text. Evidence: "PIQA [BZB <sup>+</sup> 20]" and "Physical commonsense" (Table 1).
- BoolQ - Task type: Classification; Reasoning / relational. Dataset(s): BoolQ [CLC+19]. Domain: text. Evidence: "BoolQ [CLC+19]" and "Question answering" (Table 1).
- CB - Task type: Classification; Reasoning / relational. Dataset(s): CB [dMST19]. Domain: text. Evidence: "CB [dMST19]" and "Textual entailment" (Table 1).
- COPA - Task type: Classification; Reasoning / relational. Dataset(s): COPA [RBG11]. Domain: text. Evidence: "COPA [RBG11]" and "Causal reasoning" (Table 1).
- Rendered SST-2 - Task type: Classification. Dataset(s): Rendered SST-2 [RKH <sup>+</sup> 21]. Domain: rendered text images. Evidence: "Rendered SST-2 [RKH <sup>+</sup> 21]" and "OCR-free sentiment classification" (Table 1); "OCR-free language understanding is a task that focuses on understanding text and images without relying on Optical Character Recognition (OCR)." (Section 4.3).
- HatefulMemes - Task type: Classification. Dataset(s): HatefulMemes [KFM <sup>+</sup> 20]. Domain: multimodal memes (image + text). Evidence: "HatefulMemes [KFM <sup>+</sup> 20]" and "OCR-free meme classification" (Table 1); "OCR-free language understanding is a task that focuses on understanding text and images without relying on Optical Character Recognition (OCR)." (Section 4.3).
- RelativeSize - Task type: Classification; Reasoning / relational. Dataset(s): RelativeSize [BHCF16]. Domain: text-only prompts about object properties. Evidence: "RelativeSize [BHCF16]" and "Commonsense reasoning (object size)" (Table 1); "We use only text as our input and do not include any images." (Section 4.9.2).
- MemoryColor - Task type: Classification; Reasoning / relational. Dataset(s): MemoryColor [NHJ21]. Domain: text-only prompts about object properties. Evidence: "MemoryColor [NHJ21]" and "Commonsense reasoning (object color)" (Table 1); "We use only text as our input and do not include any images." (Section 4.9.2).
- ColorTerms - Task type: Classification; Reasoning / relational. Dataset(s): ColorTerms [BBBT12]. Domain: text-only prompts about object properties. Evidence: "ColorTerms [BBBT12]" and "Commonsense reasoning (object color)" (Table 1); "We use only text as our input and do not include any images." (Section 4.9.2).
- IQ Test (Raven's Progressive Matrices) - Task type: Reasoning / relational; Classification. Dataset(s): IQ Test (Raven's Progressive Matrices). Domain: nonverbal image matrices. Evidence: "IQ Test" and "Raven's Progressive Matrices" (Table 1); "Given eight images presented in a  $3 \times 3$  matrix, the task is to identify the following element from six similar candidates." (Section 4.2).
- COCO Caption - Task type: Generation. Dataset(s): COCO Caption [LMB <sup>+</sup> 14]. Domain: natural images. Evidence: "COCO Caption [LMB <sup>+</sup> 14]" and "Image captioning" (Table 1); "Image captioning involves generating a natural language description of an image" (Section 4.1).
- Flicker30k - Task type: Generation. Dataset(s): Flicker30k [YLHH14]. Domain: natural images. Evidence: "Flicker30k [YLHH14]" and "Image captioning" (Table 1); "Image captioning involves generating a natural language description of an image" (Section 4.1).
- VQAv2 - Task type: Generation; Reasoning / relational. Dataset(s): VQAv2 [GKSS <sup>+</sup> 17]. Domain: natural images. Evidence: "VQAv2 [GKSS <sup>+</sup> 17]" and "Visual question answering" (Table 1); "visual question answering aims to answer a natural language question with respect to an image." (Section 4.1).
- VizWiz - Task type: Generation; Reasoning / relational. Dataset(s): VizWiz [GLS <sup>+</sup> 18]. Domain: natural images. Evidence: "VizWiz [GLS <sup>+</sup> 18]" and "Visual question answering" (Table 1); "visual question answering aims to answer a natural language question with respect to an image." (Section 4.1).
- WebSRC - Task type: Generation; Reasoning / relational. Dataset(s): WebSRC [CZC <sup>+</sup> 21]. Domain: web page images (with extracted text in prompt). Evidence: "WebSRC [CZC <sup>+</sup> 21]" and "Web page question answering" (Table 1); "Web page question answering aims at finding answers to questions from web pages." (Section 4.4).
- ImageNet - Task type: Classification. Dataset(s): ImageNet [DDS+09]. Domain: natural images. Evidence: "ImageNet [DDS+09]" and "Zero-shot image classification" (Table 1); "Image classification comprehends an entire image as a whole and aims to assign a label to the image." (Section 4.6).
- CUB (classification with descriptions) - Task type: Classification. Dataset(s): CUB [WBW <sup>+</sup> 11]. Domain: natural images with category descriptions. Evidence: "CUB [WBW <sup>+</sup> 11]" and "Zero-shot image classification with descriptions" (Table 1); "Following CUB [WBW+11], we construct a bird classification dataset that contains images and natural-language descriptions of categories." (Section 4.7.1).

## 4. Domain and Modality Scope

- Single domain? No; the evaluation spans multiple task families: "language, perception-language, and vision tasks." (Introduction).
- Multiple domains within the same modality? Yes; multiple language-task domains are evaluated: "We evaluate KOSMOS-1 and the LLM baseline on eight language tasks, including cloze and completion tasks (i.e, StoryCloze, HellaSwag), Winograd-style tasks (i.e, Winograd, Winogrande), commonsense reasoning (i.e, PIQA), and three datasets BoolQ, CB, and COPA from the SuperGLUE benchmark [WPN<sup>+</sup>19]." (Section 4.8.1).
- Multiple modalities? Yes: "we train KOSMOS-1 from scratch on web-scale multimodal corpora, including arbitrarily interleaved text and images, image-caption pairs, and text data." (Abstract).
- Domain generalization or cross-domain transfer? Cross-modal transfer is claimed: "We also show that MLLMs can benefit from cross-modal transfer, i.e., transfer knowledge from language to multimodal, and from multimodal to language." (Abstract). Domain generalization is not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| StoryCloze | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| HellaSwag | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| Winograd | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| Winogrande | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| PIQA | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| BoolQ | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| CB | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| COPA | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| Rendered SST-2 | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| HatefulMemes | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| RelativeSize | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| MemoryColor | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| ColorTerms | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| IQ Test (Raven's Progressive Matrices) | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| COCO Caption | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| Flicker30k | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| VQAv2 | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| VizWiz | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| WebSRC | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| ImageNet | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |
| CUB (classification with descriptions) | Yes (single model evaluated across tasks) | No (zero-/few-shot only) | Not specified | "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract) |

## 6. Input and Representation Constraints

- Sequence flattening and special tokens: "For input format, we flatten input as a sequence decorated with special tokens. Specifically, we use <s> and </s> to denote start-and end-of-sequence. The special tokens <image> and </image> indicate the beginning and end of encoded image embeddings." (Section 2.1).
- Vision encoder and pooling: "we employ a vision encoder as the embedding module for input images. In addition, Resampler [ADL<sup>+</sup>22] is used as an attentive pooling mechanism to reduce the number of image embeddings." (Section 2.1).
- Fixed image resolution (training): "The images are preprocessed into 224×224 resolution during training." (Section 3.2).
- Fixed image resolution (evaluation examples): "The image resolution is 224×224." (Section 4.1.1).
- Maximum sequence length: "Max length                  | 2,048" (Table 17); "Max length of text corpora        | 2,048" (Table 18).
- Input image count limit in interleaved data: "For each document, we limit the number of images to five to reduce noise and redundancy." (Section 3.1).
- Minimum image resolution filtering: "we discard any images that have a resolution lower than 64 by 64 pixels or that are single-colored." (Section B.1.3).
- Fixed patch size: Not specified.
- Fixed number of tokens: Maximum length is specified; a fixed token count per example is not specified.
- Fixed dimensionality (strictly 2D): Not explicitly specified beyond 224x224 preprocessing for images.
- Padding or resizing requirements: Resizing implied by "preprocessed into 224×224 resolution" (Section 3.2).

## 7. Context Window and Attention Structure

- Maximum sequence length: "Max length                  | 2,048" (Table 17); "Max length of text corpora        | 2,048" (Table 18).
- Fixed vs variable length: A fixed maximum length is specified; variable-length handling is not explicitly stated.
- Attention type: Causal self-attention is described for the Transformer decoder: "The left-to-right causal model processes the sequence in an auto-regressive manner" and "The causal masking is used to mask out future information." (Section 2.2).
- Computational cost mechanisms: "Resampler [ADL<sup>+</sup>22] is used as an attentive pooling mechanism to reduce the number of image embeddings." (Section 2.1).

## 8. Positional Encoding (Critical Section)

- Mechanism: Relative positional encoding (xPos) is used: "We employ xPos [SDP<sup>+</sup>22] relative position encoding for better long-context modeling." (Section 2.2) and "Relative position embedding | $xPos [SDP^+22]$" (Table 17).
- Where applied: Not specified.
- Fixed across experiments vs modified per task: Not specified; no task-specific positional encoding changes are described.
- Ablated or compared against alternatives: Not described.

## 9. Positional Encoding as a Variable

- Positional encoding is treated as a fixed architectural choice: "We employ xPos [SDP<sup>+</sup>22] relative position encoding for better long-context modeling." (Section 2.2).
- Multiple positional encodings compared? Not reported.
- Claims that PE is "not critical" or secondary? Not stated.

## 10. Evidence of Constraint Masking

- Model size: "The MLLM component has 24 layers with 2,048 hidden dimensions, 8,192 FFN intermediate size, and 32 attention heads, resulting in about 1.3B parameters." and "The total number of parameters of KOSMOS-1 is about 1.6B." (Section 3.2).
- Training data scale: "train Kosmos-1 for 300k steps, corresponding to about 360 billion tokens." (Section 3.2).
- Interleaved web data scale: "we select about 71M web pages from the original 2B web pages in the snapshot." (Section 3.1).
- Image-caption pair scale: "LAION-2B contains about 2B English image-caption pairs, LAION-400M consists of 400M English image-caption pairs, and COYO-700M has 700M English image-caption pairs." (Section B.1.2).
- Training-trick attribution: "Language-only instruction tuning boosts our model's performance by 1.9 points on Flickr30k, 4.3 points on VQAv2, and 1.3 points on VizWiz." (Section 4.9.1).
- Scaling attribution: The architecture notes scaling stability but does not attribute gains primarily to scaling: "The method has a theoretically derived initialization method [WMD<sup>+</sup>22] to improve the optimization fundamentally, which allows us to effectively scale up the models without pain." (Section 2.2). Performance gains are not explicitly credited to model size or data scaling beyond these statements.

## 11. Architectural Workarounds

- Attentive pooling to reduce image tokens: "Resampler [ADL<sup>+</sup>22] is used as an attentive pooling mechanism to reduce the number of image embeddings." (Section 2.1).
- Transformer variant for stability and scale: "We use MAGNETO [WMH<sup>+</sup>22], a Transformer variant, as the backbone architecture. MAGNETO has better training stability and superior performance across modalities. It introduces an extra LayerNorm to each sublayer" and "allows us to effectively scale up the models without pain." (Section 2.2).
- Long-context positional encoding: "We employ xPos [SDP<sup>+</sup>22] relative position encoding for better long-context modeling." (Section 2.2).
- Frozen vision encoder to manage training: "the image representation is obtained from a pretrained CLIP ViT-L/14 model" and "We freeze the parameters of the CLIP model except for the last layer during training." (Section 3.2).

## 12. Explicit Limitations and Non-Claims

- Limitation: "Although there is still a large performance gap between the current model and the average level of adults" (Section 4.2.2).
- Future work: "In the future, we would like to scale up KOSMOS-1 in terms of model size" and "integrate the speech [WCW<sup>+</sup>23] capability into KOSMOS-1." (Section 5 Conclusion).
- Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)

**Constraint Profile:**
- Domain scope: Multiple modalities and domains (language, perception-language, vision) with fixed benchmark datasets.
- Task structure: Many supervised-style benchmarks evaluated in zero-/few-shot prompting with no per-task training.
- Representation rigidity: Fixed tokenization and explicit image embedding boundaries; fixed 224x224 image preprocessing and max length 2,048.
- Model sharing vs specialization: Single shared model across tasks; no per-task finetuning.
- Role of positional encoding: Relative xPos used as a fixed architectural component, not a research variable.

### 14. Final Classification

**Multi-task, multi-domain (constrained).** The evaluation spans "language, perception-language, and vision tasks" and uses specific benchmarks (Table 1), indicating multiple domains and modalities. At the same time, tasks are evaluated in zero-/few-shot prompting "without any gradient updates or finetuning," so the setup is broad but constrained to fixed datasets and prompting protocols rather than open-ended multi-domain learning.
