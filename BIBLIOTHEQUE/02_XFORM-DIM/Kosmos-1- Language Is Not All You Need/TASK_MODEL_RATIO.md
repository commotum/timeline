Number of distinct tasks evaluated: 17.

- "| StoryCloze [MRL+17]                  | Commonsense reasoning                            | Accuracy    | ✓         | ✓        |" (Table 1)
- "| HellaSwag [ZHB+19]                   | Commonsense NLI                                  | Accuracy    | /         | 1        |" (Table 1)
- "| Winograd [LDM12a]                    | Word ambiguity                                   | Accuracy    | ✓         | /        |" (Table 1)
- "| PIQA [BZB <sup>+</sup> 20]           | Physical commonsense                             | Accuracy    | ✓         | ✓        |" (Table 1)
- "| BoolQ [CLC+19]                       | Question answering                               | Accuracy    | ✓         | /        |" (Table 1)
- "| CB [dMST19]                          | Textual entailment                               | Accuracy    | ✓         | /        |" (Table 1)
- "| COPA [RBG11]                         | Causal reasoning                                 | Accuracy    | ✓         | ✓        |" (Table 1)
- "| Rendered SST-2 [RKH <sup>+</sup> 21] | OCR-free sentiment classification                | Accuracy    | ✓         |          |" (Table 1)
- "| HatefulMemes [KFM <sup>+</sup> 20]   | OCR-free meme classification                     | ROC AUC     | ✓         |          |" (Table 1)
- "| RelativeSize [BHCF16]                | Commonsense reasoning (object size)              | Accuracy    | ✓         |          |" (Table 1)
- "| MemoryColor [NHJ21]                  | Commonsense reasoning (object color)             | Accuracy    | ✓         |          |" (Table 1)
- "| IQ Test                              | Raven's Progressive Matrices                     | Accuracy    | ✓         |          |" (Table 1)
- "| COCO Caption [LMB <sup>+</sup> 14]   | Image captioning                                 | CIDEr, etc. | ✓         | ✓        |" (Table 1)
- "| VQAv2 [GKSS <sup>+</sup> 17]         | Visual question answering                        | VQA acc.    | ✓         | ✓        |" (Table 1)
- "| WebSRC [CZC <sup>+</sup> 21]         | Web page question answering                      | F1 score    | ✓         |          |" (Table 1)
- "| ImageNet [DDS+09]                    | Zero-shot image classification                   | Top-1 acc.  | ✓         |          |" (Table 1)
- "| CUB [WBW <sup>+</sup> 11]            | Zero-shot image classification with descriptions | Accuracy    | ✓         |          |" (Table 1)

Number of trained model instances required to cover all tasks: 1.

- "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract)

$$
\boxed{
\frac{17\ \text{tasks}}{1\ \text{model}} = 17
}
$$
