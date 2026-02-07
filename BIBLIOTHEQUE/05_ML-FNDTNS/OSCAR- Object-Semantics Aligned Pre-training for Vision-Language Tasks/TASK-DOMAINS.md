# Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks (Not specified in the paper.)
Source: OSCAR- Object-Semantics Aligned Pre-training for Vision-Language Tasks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image retrieval | image-text pair (image regions + text) | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | alignment probability score (used to rank images) | 0D (inferred) | Fixed (inferred) |
| Text retrieval | image-text pair (image regions + text) | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | alignment probability score (used to rank captions) | 0D (inferred) | Fixed (inferred) |
| Image captioning | image regions + object tags | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | caption tokens (natural language description) | 1D (t) (inferred) | Not specified in the paper. |
| Novel object captioning (NoCaps) | image regions + object tags (Open Images labels) | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | caption tokens describing novel objects | 1D (t) (inferred) | Not specified in the paper. |
| Visual question answering (VQA) | image + question (with object tags and region features) | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | answer label from multi-choice list | 0D (inferred) | Fixed (inferred) |
| GQA | image + question | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | answer label from candidate set | 0D (inferred) | Fixed (inferred) |
| NLVR2 | pair of images + natural language statement | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | binary truth label | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates OSCAR on seven vision-language tasks spanning understanding (image/text retrieval, VQA, GQA, NLVR2) and generation (image captioning, NoCaps). Inputs combine 2D images and 1D text, while outputs are 0D labels/scores for understanding tasks and 1D text sequences for captioning. Input dynamics are capped by fixed token/region sequence lengths (inferred), attention is static via fixed self-attention patterns (inferred), and state is constructed through fused [CLS] representations (inferred); caption output dynamics are not specified.

## Evidence
### Task: Image retrieval
- "There are two sub-tasks:  $image\ retrieval$  and  $text\ retrieval$ , depending on which modality is used as the retrieved target." (Section 4 Adapting to V+L Tasks)
- "In the testing stage, the probability score is used to rank the given image-text pairs of a query." (Section 4 Adapting to V+L Tasks)
- "During training, we formulate it as a binary classification problem." (Section 4 Adapting to V+L Tasks)
- Inference: In/Out Dimension labels and capped inputs are inferred from "Given an aligned image-text pair" (Section 4 Adapting to V+L Tasks) and "The sequence length of discrete tokens  $\boldsymbol{h}$  and region features  $\boldsymbol{v}$  are 35 and 50, respectively." (Section 3 Oscar Pre-training). Static attention is inferred from "The default setting uses full attentions across all modalities." (Section 5.3 Ablation Analysis). Constructed state is inferred from "the encoder output on the special token [CLS] is the fused vision-language representation" (Section 3 Oscar Pre-training).

### Task: Text retrieval
- "There are two sub-tasks:  $image\ retrieval$  and  $text\ retrieval$ , depending on which modality is used as the retrieved target." (Section 4 Adapting to V+L Tasks)
- "In the testing stage, the probability score is used to rank the given image-text pairs of a query." (Section 4 Adapting to V+L Tasks)
- "During training, we formulate it as a binary classification problem." (Section 4 Adapting to V+L Tasks)
- Inference: In/Out Dimension labels and capped inputs are inferred from "Given an aligned image-text pair" (Section 4 Adapting to V+L Tasks) and "The sequence length of discrete tokens  $\boldsymbol{h}$  and region features  $\boldsymbol{v}$  are 35 and 50, respectively." (Section 3 Oscar Pre-training). Static attention is inferred from "The default setting uses full attentions across all modalities." (Section 5.3 Ablation Analysis). Constructed state is inferred from "the encoder output on the special token [CLS] is the fused vision-language representation" (Section 3 Oscar Pre-training).

### Task: Image captioning
- "Image Captioning requires the model to generate a natural language description of the content of an image." (Section 4 Adapting to V+L Tasks)
- "During inference, we first encode the image regions, object tags, and a special token [CLS] as input." (Section 4 Adapting to V+L Tasks)
- "self-attention mask is constrained such that a caption token can only attend to the tokens before its position" (Section 4 Adapting to V+L Tasks)
- Inference: In/Out Dimension labels and capped inputs are inferred from "The sequence length of discrete tokens  $\boldsymbol{h}$  and region features  $\boldsymbol{v}$  are 35 and 50, respectively." (Section 3 Oscar Pre-training). Constructed state is inferred from "the encoder output on the special token [CLS] is the fused vision-language representation" (Section 3 Oscar Pre-training). Out dynamics are not specified for captions.

### Task: Novel object captioning (NoCaps)
- "Novel Object Captioning (NoCaps) [1] extends the image captioning task" (Section 4 Adapting to V+L Tasks)
- "provides a benchmark with images from the Open Images dataset [17] to test models' capability of describing novel objects" (Section 4 Adapting to V+L Tasks)
- "we use the predicted Visual Genome and Open Images labels to form tag sequences" (Section 4 Adapting to V+L Tasks)
- Inference: In/Out Dimension labels and capped inputs are inferred from "The sequence length of discrete tokens  $\boldsymbol{h}$  and region features  $\boldsymbol{v}$  are 35 and 50, respectively." (Section 3 Oscar Pre-training). Static attention is inferred from the captioning attention description: "self-attention mask is constrained such that a caption token can only attend to the tokens before its position" (Section 4 Adapting to V+L Tasks). Constructed state is inferred from "the encoder output on the special token [CLS] is the fused vision-language representation" (Section 3 Oscar Pre-training). Out dynamics are not specified for captions.

### Task: Visual question answering (VQA)
- "VQA [9] requires the model to answer natural language questions based on an image." (Section 4 Adapting to V+L Tasks)
- "Given an image and a question, the task is to select the correct answer from a multi-choice list." (Section 4 Adapting to V+L Tasks)
- "we construct one input sequence, which contains the concatenation of a given question, object tags and region features" (Section 4 Adapting to V+L Tasks)
- Inference: In/Out Dimension labels and capped inputs are inferred from "The sequence length of discrete tokens  $\boldsymbol{h}$  and region features  $\boldsymbol{v}$  are 35 and 50, respectively." (Section 3 Oscar Pre-training). Static attention is inferred from "relies on the self-attention mechanism to learn image-text alignments" (Section 2 Background). Constructed state is inferred from "the [CLS] output from OSCAR is fed to a task-specific linear classifier for answer prediction." (Section 4 Adapting to V+L Tasks).

### Task: GQA
- "**GQA** [13] is similar to VQA, except that GQA tests the reasoning capability of the model to answer a question." (Section 4 Adapting to V+L Tasks)
- "VQA [9] requires the model to answer natural language questions based on an image." (Section 4 Adapting to V+L Tasks)
- "For each question, the model chooses an answer from a shared set of 1,852 candidate answers." (Section 4 Adapting to V+L Tasks)
- Inference: In/Out Dimension labels and capped inputs are inferred from "The sequence length of discrete tokens  $\boldsymbol{h}$  and region features  $\boldsymbol{v}$  are 35 and 50, respectively." (Section 3 Oscar Pre-training). Static attention is inferred from "relies on the self-attention mechanism to learn image-text alignments" (Section 2 Background). Constructed state is inferred from "the encoder output on the special token [CLS] is the fused vision-language representation" (Section 3 Oscar Pre-training).

### Task: NLVR2
- "Natural Language Visual Reasoning for Real (NLVR2) [36] takes a pair of images and a natural language," (Section 4 Adapting to V+L Tasks)
- "the goal is to determine whether the natural language statement is true about the image pair." (Section 4 Adapting to V+L Tasks)
- "two [CLS] outputs from OSCAR are concatenated as the joint input for a binary classifier" (Section 4 Adapting to V+L Tasks)
- Inference: In/Out Dimension labels and capped inputs are inferred from "The sequence length of discrete tokens  $\boldsymbol{h}$  and region features  $\boldsymbol{v}$  are 35 and 50, respectively." (Section 3 Oscar Pre-training). Static attention is inferred from "relies on the self-attention mechanism to learn image-text alignments" (Section 2 Background). Constructed state is inferred from "two [CLS] outputs from OSCAR are concatenated as the joint input for a binary classifier" (Section 4 Adapting to V+L Tasks).
