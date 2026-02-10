# Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision (2020)
Source: Vokenization- Improving Language Understanding with Contextualized, Visual-Grounded Supervision.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Masked language modeling | Masked token sequence | 1D (t) (inferred) | Fixed | Static (inferred) | Direct (inferred) | Predicted masked tokens | 1D (t) (inferred) | Capped (inferred) |
| Voken classification | Token sequence (language context) | 1D (t) (inferred) | Fixed | Static (inferred) | Direct (inferred) | Voken label per token (from finite image set) | 1D (t) (inferred) | Fixed (inferred) |
| Contextual token-image relevance scoring | Token-in-sentence and image pair | 1D (t); 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Relevance score for token-image pair | 0D (inferred) | Fixed (inferred) |
| Vokenization (token-to-image retrieval) | Sentence tokens and candidate image set | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Direct (inferred) | Retrieved image (voken) per token | 1D (t); 2D (x, y) (inferred) | Capped (inferred) |
| GLUE text classification/inference | Sentence or sentence-pair tokens | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Task label (e.g., sentiment, entailment, paraphrase, NLI) | 0D (inferred) | Fixed (inferred) |
| Extractive question answering (SQuAD) | Question and passage tokens | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Answer span/text | 1D (t) (inferred) | Capped (inferred) |
| Multiple-choice commonsense inference (SWAG) | Context plus candidate continuation tokens | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Selected choice label | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers both multimodal pre-training tasks and downstream pure-language tasks. In pre-training, it combines masked language modeling with token-level voken classification, and it builds vokens through contextual token-image scoring and retrieval over a finite image set. The task space spans 1D text streams and 2D images, with predominantly Fixed or Capped dynamics where explicitly supported (fixed sequence length 128, finite voken set). Attention is mostly Static for encoder-based prediction, while voken retrieval is Dynamic because the system selects images at runtime.

## Evidence
### Task: Masked language modeling
- "The original BERT pre-training mainly relies on the task of masked language model" (Section 2.2)
- "the model needs to predict these missing tokens from language context." (Section 2.2)
- "Using fixed sequence length (Conneau et al., 2020) of 128." (Section 4.2)
- Inference: `1D (t)` input/output dimensions are inferred from the token sequence notation `s = {w_i}` in Section 2.2. `Static` attention and `Direct` state are inferred because this objective consumes a fixed input slice and predicts tokens without any explicit retrieval or maintained external state. `Capped` output dynamics are inferred because predictions are only over the masked subset `\hat{s}`.

### Task: Voken classification
- "Based on these vokens, we propose a new pre-training task for language: voken classification." (Section 2.1)
- "Suppose the vokens come from a finite set  $\mathbb{X}$" (Section 2.2)
- "the voken classification loss is the negative log probability of all corresponding vokens:" (Section 2.2)
- Inference: `1D (t)` dimensions are inferred because the objective is token-level over sequence positions. `Static` attention and `Direct` state are inferred because the model predicts from the given token context without runtime retrieval during this objective. `Fixed` output dynamics are inferred from token-level prediction under the fixed-length pre-training setup in Section 4.2.

### Task: Contextual token-image relevance scoring
- "The model takes a sentence s and an image x as input" (Section 3.2)
- "The output  $r_{\theta}(w_i, x; s)$  is the relevance score between the token  $w_i \in s$  and the image x while considering the whole sentence s as a context." (Section 3.2)
- "we factorize it as an inner product of the language feature representation  $f_{\theta}(w_i; s)$  and the visual feature representation  $g_{\theta}(x)$ :" (Section 3.2)
- Inference: `1D (t); 2D (x, y)` input dimension is inferred from sentence tokens plus image input. `0D` output is inferred from the scalar "relevance score." `Static` attention and `Direct` state are inferred because scoring is computed directly from the provided pair without runtime retrieval policy or persistent constructed memory.

### Task: Vokenization (token-to-image retrieval)
- "As shown in Fig. 1 and Fig. 2, vokenization is the process to assign each token  $w_i$  in a sentence  $s = (w_1, w_2, \dots, w_l)$  with a relevant image  $v(w_i; s)$ ." (Section 3.1)
- "Instead of creating this image with generative models, we retrieve an image from a set of images  $\mathbb{X} = \{x_1, x_2, \dots, x_n\}$" (Section 3.1)
- "We fix a voken size of 50000." (Section 4.2)
- Inference: `1D (t); 2D (x, y)` dimensions are inferred because the output is a token-indexed sequence of retrieved images. `Dynamic` attention is inferred from runtime selection via `argmax` over candidate images. `Capped` dynamics are inferred from the finite candidate set (explicitly fixed to 50,000 images).

### Task: GLUE text classification/inference
- "The pre-trained models are then fine-tuned on GLUE (Wang et al., 2019), SQuAD (Rajpurkar et al., 2016, 2018), and SWAG (Zellers et al., 2018) to assess the pre-training performance." (Section 4.1)
- "We follow this trend and evaluate on the four largest datasets (i.e., SST-2 (Socher et al., 2013), QNLI (Rajpurkar et al., 2016), QQP (Iyer et al., 2017), MNLI (Williams et al., 2018))." (Section 4.1)
- "The default metric is accuracy." (Section 4.3)
- Inference: `1D (t)` input is inferred from sentence/sentence-pair NLU benchmarks. `0D` output with `Fixed` dynamics is inferred from classification-style evaluation (accuracy) and finite label spaces in these tasks. `Static` attention and `Direct` state are inferred because fine-tuning uses the encoder as a direct mapping from text inputs to task labels.

### Task: Extractive question answering (SQuAD)
- "The pre-trained models are then fine-tuned on GLUE (Wang et al., 2019), SQuAD (Rajpurkar et al., 2016, 2018), and SWAG (Zellers et al., 2018) to assess the pre-training performance." (Section 4.1)
- "For SQuAD, we report the exact matching and F1 score respectively." (Section 4.3)
- "SQuAD results are \"exact match\"/\"F1\"." (Table 2 caption)
- Inference: `1D (t)` input and output are inferred from question-context token sequences and span/text answers. `Capped` output dynamics are inferred because answer spans are bounded by the provided context. `Static` attention and `Direct` state are inferred from encoder-style fine-tuning without explicit runtime retrieval modules described for this stage.

### Task: Multiple-choice commonsense inference (SWAG)
- "The pre-trained models are then fine-tuned on GLUE (Wang et al., 2019), SQuAD (Rajpurkar et al., 2016, 2018), and SWAG (Zellers et al., 2018) to assess the pre-training performance." (Section 4.1)
- "The hyper-parameters for SQuAD, SWAG are borrowed from BERT." (Section 4.2)
- "The default metric is accuracy." (Section 4.3)
- Inference: `1D (t)` input is inferred from textual context and candidate endings. `0D` output with `Fixed` dynamics is inferred from single-label multiple-choice prediction evaluated by accuracy. `Static` attention and `Direct` state are inferred because prediction is performed from the provided textual options without an explicit runtime retrieval controller in this stage.
