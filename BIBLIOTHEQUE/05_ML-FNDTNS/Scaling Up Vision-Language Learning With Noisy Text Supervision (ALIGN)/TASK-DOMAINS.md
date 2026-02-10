# Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision (2021)
Source: Scaling Up Vision-Language Learning With Noisy Text Supervision (ALIGN).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image-to-text classification (contrastive pre-training) | Images and candidate texts (in-batch) | 2D (x, y) (inferred); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Matched text class | 0D (inferred) | Capped (inferred) |
| Text-to-image classification (contrastive pre-training) | Texts and candidate images (in-batch) | 1D (t) (inferred); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Matched image class | 0D (inferred) | Capped (inferred) |
| Image-to-text retrieval | Images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Retrieved texts | 1D (t) (inferred) | Capped (inferred) |
| Text-to-image retrieval | Texts | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Retrieved images | 2D (x, y) (inferred) | Capped (inferred) |
| Text-to-text retrieval | Text pairs or text queries | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Retrieved texts | 1D (t) (inferred) | Capped (inferred) |
| Image-to-image retrieval | Image pairs or image queries | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Retrieved images | 2D (x, y) (inferred) | Capped (inferred) |
| Semantic textual similarity (STS) | Text pairs | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Similarity score | 0D (inferred) | Fixed (inferred) |
| Semantic image similarity (SIS) | Image pairs | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Similarity score | 0D (inferred) | Fixed (inferred) |
| Semantic image-text similarity (SITS) | Image-text pairs | 2D (x, y) (inferred); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Similarity score | 0D (inferred) | Fixed (inferred) |
| Visual classification | Images; class-name texts (zero-shot setting) | 2D (x, y) (inferred); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Class labels | 0D (inferred) | Fixed (inferred) |
| Image retrieval with multimodal query (image+text) | Query image and text string | 2D (x, y) (inferred); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Retrieved images | 2D (x, y) (inferred) | Capped (inferred) |
| Word similarity estimation (SimLex-999) | Word pairs (wordpiece tokens) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Similarity score | 0D (inferred) | Fixed (inferred) |

## Summary
ALIGN covers contrastive cross-modal matching, cross-modal and intra-modal retrieval, semantic similarity scoring, visual classification, multimodal-query image search, and word-level similarity evaluation. The paper supports 2D (x, y) image inputs and 1D (t) token inputs, including mixed multimodal inputs, with outputs spanning retrieved texts/images and 0D labels or similarity scores. From the model description, image inputs are typically fixed-resolution while text and retrieval candidate sets are capped, so dynamics are mostly Fixed or Capped rather than Open. The dual-encoder setup with learned embeddings supports Static attention and Constructed state assignments across these tasks.

## Evidence
### Task: Image-to-text classification (contrastive pre-training)
- "We minimize the sum of two losses: one for image-to-text classification" (Section 4.1)
- "In training, we treat matched image-text pairs as positive and all other random image-text pairs that can be formed in a training batch as negative." (Section 4.1)
- Inference: In Dimension/In Dynamics, Attention Dynamic, State Dynamic, Out Dimension, and Out Dynamics are inferred from "The model consists of a pair of image and text encoders with a cosine-similarity combination function at the top." (Section 4.1) and "For BERT we use wordpiece sequence of maximum 64 tokens" plus "The image encoder is trained at resolution of  $289 \times 289$  pixels" (Section 5).

### Task: Text-to-image classification (contrastive pre-training)
- "and the other for text-to-image classification" (Section 4.1)
- "In training, we treat matched image-text pairs as positive and all other random image-text pairs that can be formed in a training batch as negative." (Section 4.1)
- Inference: In Dimension/In Dynamics, Attention Dynamic, State Dynamic, Out Dimension, and Out Dynamics are inferred from the same encoder/interface evidence in Sections 4.1 and 5 (fixed image tower + capped text tokens + dual-encoder embedding matching).

### Task: Image-to-text retrieval
- "We evaluate ALIGN models on image-to-text and text-toimage retrieval tasks, with and without finetuning." (Section 4.2)
- "Table 1. Image-text retrieval results on Flickr30K and MSCOCO datasets (zero-shot and fine-tuned)." (Section 5.1)
- Inference: In/Out Dimension and dynamics labels are inferred from the modality definition (image input and text output), fixed image preprocessing in Section 5, and capped ranked retrieval behavior evidenced by Recall@K reporting.

### Task: Text-to-image retrieval
- "We evaluate ALIGN models on image-to-text and text-toimage retrieval tasks, with and without finetuning." (Section 4.2)
- "Figure 4 shows the top 1 text-to-image retrieval results for a handful of text queries not existing in the training data." (Section 7)
- Inference: In/Out Dimension and dynamics labels are inferred from text-query to image-result retrieval, capped text length ("maximum 64 tokens" in Section 5), and ranked retrieval outputs (top-1/Recall@K).

### Task: Text-to-text retrieval
- "With extended annotations, CxC enables four intra- and inter-modal retrieval tasks including image-to-text, text-toimage, text-to-text, and image-to-image retrieval" (Section 4.2)
- "the improvements on text-to-text and image-to-image retrieval tasks (in particular the former) are less significant" (Section 5.1)
- Inference: Dimension/dynamics and attention/state labels are inferred from text retrieval over encoded text embeddings under the same dual-encoder static interface.

### Task: Image-to-image retrieval
- "With extended annotations, CxC enables four intra- and inter-modal retrieval tasks including image-to-text, text-toimage, text-to-text, and image-to-image retrieval" (Section 4.2)
- "the improvements on text-to-text and image-to-image retrieval tasks (in particular the former) are less significant" (Section 5.1)
- Inference: Dimension/dynamics and attention/state labels are inferred from image retrieval over fixed-resolution image embeddings and ranked outputs.

### Task: Semantic textual similarity (STS)
- "With extended annotations, CxC enables four intra- and inter-modal retrieval tasks including image-to-text, text-toimage, text-to-text, and image-to-image retrieval, and three semantic similarity tasks including semantic textual similarity (STS), semantic image similarity (SIS), and semantic image-text similarity (SITS)." (Section 4.2)
- "Table 3. Spearman's R Bootstrap Correlation ( $\times 100$ ) on Crisscrossed Captions (CxC) dataset." (Section 5.1)
- Inference: Out Dimension/Out Dynamics are inferred as scalar similarity scoring from the correlation-based evaluation setup; input/dynamics follow capped text encoder constraints.

### Task: Semantic image similarity (SIS)
- "With extended annotations, CxC enables four intra- and inter-modal retrieval tasks including image-to-text, text-toimage, text-to-text, and image-to-image retrieval, and three semantic similarity tasks including semantic textual similarity (STS), semantic image similarity (SIS), and semantic image-text similarity (SITS)." (Section 4.2)
- "The performance on STS and SIS tasks is also slightly worse than VSE++" (Section 5.1)
- Inference: Out Dimension/Out Dynamics are inferred as scalar similarity scoring, while 2D image input with fixed preprocessing is inferred from the model/evaluation pipeline.

### Task: Semantic image-text similarity (SITS)
- "With extended annotations, CxC enables four intra- and inter-modal retrieval tasks including image-to-text, text-toimage, text-to-text, and image-to-image retrieval, and three semantic similarity tasks including semantic textual similarity (STS), semantic image similarity (SIS), and semantic image-text similarity (SITS)." (Section 4.2)
- "Table 3 shows that ALIGN also outperforms the previous SOTA on SITS task with an improvement of 5.7%." (Section 5.1)
- Inference: Mixed image+text input dimensions and scalar similarity output are inferred from the task definition and correlation-style evaluation.

### Task: Visual classification
- "We first apply zero-shot transfer of ALIGN to visual classification tasks on ImageNet ILSVRC-2012 benchmark" (Section 4.3)
- "If we directly feed the texts of classnames into the text encoder, ALIGN is able to classify images into candidate classes via image-text retrieval." (Section 5.2)
- Inference: Mixed input dimensions (image plus class-name text in zero-shot mode), Capped input dynamics, and 0D label outputs are inferred from the zero-shot classification procedure and fixed image/text encoder interfaces.

### Task: Image retrieval with multimodal query (image+text)
- "Without any fine-tuning, ALIGN powers zero-shot visual classification and cross-modal search including image-to-text search, text-to-image search and even search with joint image+text queries." (Figure 1 caption / Introduction)
- "Specifically, given a query image and a text string, we add their ALIGN embeddings together and use it to retrieve relevant images." (Section 7)
- Inference: Mixed 2D+1D input dimensions, capped dynamics, and image retrieval output are inferred from the explicit query construction and ranked retrieval setup.

### Task: Word similarity estimation (SimLex-999)
- "we also evaluate the word representation from ALIGN model<sup>5</sup> on SimLex-999 (Hill et al., 2015), which is a task to compare word similarity for 999 word pairs." (Appendix B)
- "As ALIGN uses the wordpiece tokens, one word can be split into multiple pieces." (Appendix B, footnote 5)
- Inference: 1D token input and capped dynamics are inferred from wordpiece-tokenized word pairs; scalar output is inferred from "SimLex-999 results (Spearman's  $\rho$ )."
