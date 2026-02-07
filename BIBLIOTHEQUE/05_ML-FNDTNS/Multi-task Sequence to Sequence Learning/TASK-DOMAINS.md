# MULTI-TASK SEQUENCE TO SEQUENCE LEARNING (Not specified in the paper)
Source: Multi-task Sequence to Sequence Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Machine translation | sequence of words (source language) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | sequence of words (target language) | 1D (t) (inferred) | Not specified in the paper. |
| Constituency parsing | sequence of English words | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | sequence of tags (linearized parse) | 1D (t) (inferred) | Not specified in the paper. |
| Image caption generation | images | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | sequence of English words (captions) | 1D (t) (inferred) | Not specified in the paper. |
| Autoencoder (sequence reconstruction) | sequence of English/German words | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | same sequence of words (reconstruction) | 1D (t) (inferred) | Not specified in the paper. |
| Skip-thought (predict related sentence half) | sequence of words (sentence or half) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | related sequence of words (next sentence/other half) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper covers sequence-to-sequence tasks spanning text-to-text generation (machine translation, constituency parsing, autoencoder reconstruction, skip-thought prediction) and image-to-text generation (image captioning). The inputs and outputs are described as sequences of words/tags and images, supporting 1D (t) and 2D (x, y) dimensions (inferred), while interface dynamics are not specified. The models are described as attention-free encoder-decoder systems, which supports Static attention and Constructed state dynamics (inferred).

## Evidence
### Task: Machine translation
- "the tasks of machine translation (MT), constituency parsing, and image caption generation." (Section 3)
- "a sequence of German words for machine translation" (Section 3.1)
- Inference: Labeled In/Out Dimension as 1D (t) because the task uses sequences of words ("sequence of German words"), labeled Attention Dynamic as Static because "our sequence to sequence models do not employ the attention mechanism" (Conclusion), and labeled State Dynamic as Constructed because "the encoder computes a representation s for each input sequence" (Section 2).

### Task: Constituency parsing
- "the tasks of machine translation (MT), constituency parsing, and image caption generation." (Section 3)
- "a sequence of tags for constituency parsing" (Section 3.1)
- Inference: Labeled In/Out Dimension as 1D (t) because the task maps word sequences to tag sequences ("sequence of tags"), labeled Attention Dynamic as Static because "our sequence to sequence models do not employ the attention mechanism" (Conclusion), and labeled State Dynamic as Constructed because "the encoder computes a representation s for each input sequence" (Section 2).

### Task: Image caption generation
- "the tasks of machine translation (MT), constituency parsing, and image caption generation." (Section 3)
- "the image caption generation task maps images to a sequence of English words" (Section 4.3.2)
- Inference: Labeled In Dimension as 2D (x, y) because the input is "images," labeled Out Dimension as 1D (t) because the output is a "sequence of English words," labeled Attention Dynamic as Static because "our sequence to sequence models do not employ the attention mechanism" (Conclusion), and labeled State Dynamic as Constructed because "the encoder computes a representation s for each input sequence" (Section 2).

### Task: Autoencoder (sequence reconstruction)
- "the same sequence of English words for autoencoders" (Section 3.1)
- "Our very first unsupervised learning task involves learning autoencoders from monolingual corpora" (Section 3.4)
- Inference: Labeled In/Out Dimension as 1D (t) because the task reconstructs a word sequence ("same sequence of English words"), labeled Attention Dynamic as Static because "our sequence to sequence models do not employ the attention mechanism" (Conclusion), and labeled State Dynamic as Constructed because "the encoder computes a representation s for each input sequence" (Section 2).

### Task: Skip-thought (predict related sentence half)
- "a related sequence of English words for the skip-thought objective" (Section 3.1)
- "we split each sentence into two halves; we then use one half to predict the other half." (Section 3.4)
- Inference: Labeled In/Out Dimension as 1D (t) because the task operates over sequences of words ("one half to predict the other half"), labeled Attention Dynamic as Static because "our sequence to sequence models do not employ the attention mechanism" (Conclusion), and labeled State Dynamic as Constructed because "the encoder computes a representation s for each input sequence" (Section 2).
