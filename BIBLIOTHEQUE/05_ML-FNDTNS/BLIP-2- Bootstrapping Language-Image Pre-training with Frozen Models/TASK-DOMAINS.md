# BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models (Not specified in the paper)
Source: BLIP-2- Bootstrapping Language-Image Pre-training with Frozen Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Contrastive similarity scoring (image-text) | Images; text | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Image-text similarity | 0D (inferred) | Not specified in the paper. |
| Binary classification (image-text match) | Images; text | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Match/unmatch label | 0D (inferred) | Not specified in the paper. |
| Text generation (image-grounded) | Images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Text | 1D (t) (inferred) | Not specified in the paper. |
| Instruction-following image-to-text generation | Images; text prompts | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Generated text | 1D (t) (inferred) | Not specified in the paper. |
| Image captioning | Images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Text descriptions (captions) | 1D (t) (inferred) | Not specified in the paper. |
| Visual question answering (VQA) | Images; questions (text) | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Answers (text) | 1D (t) (inferred) | Not specified in the paper. |
| Image-to-text retrieval | Images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Retrieved text | 1D (t) (inferred) | Not specified in the paper. |
| Text-to-image retrieval | Text | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Retrieved images | 2D (x, y) (inferred) | Not specified in the paper. |

## Summary
BLIP-2 is described as handling image-text alignment objectives (contrastive learning and matching), multiple image-to-text generation settings (image-grounded generation, instruction-following generation, and captioning), VQA, and bidirectional image-text retrieval. The inputs span images and text (questions or prompts), while outputs include generated text or scalar similarity/match decisions. From the task descriptions, image inputs map to 2D (x, y), text to 1D (t), and similarity/match outputs to 0D; dynamics, attention, and state are not explicitly specified.

## Evidence
### Task: Contrastive similarity scoring (image-text)
- "Image-Text Contrastive Learning (ITC) learns to align image representation and text representation" (Section 3.2)
- Inference: Mapped images to 2D (x, y), text to 1D (t), and similarity output to 0D per glossary based on the quoted image/text alignment description.

### Task: Binary classification (image-text match)
- "Image-Text Matching (ITM) aims to learn fine-grained alignment between image and text representation." (Section 3.2)
- "It is a binary classification task where the model is asked to predict whether an image-text pair is positive (matched) or negative (unmatched)." (Section 3.2)
- Inference: Mapped images to 2D (x, y), text to 1D (t), and the binary label to 0D per glossary based on the quoted image/text matching description.

### Task: Text generation (image-grounded)
- "Image-grounded Text Generation (ITG) loss trains the Q-Former to generate texts, given input images as the condition." (Section 3.2)
- Inference: Mapped images to 2D (x, y) and generated text to 1D (t) per glossary based on the quoted image-conditioned text generation description.

### Task: Instruction-following image-to-text generation
- "BLIP-2 can be prompted to perform zero-shot image-to-text generation that follows natural language instructions" (Introduction)
- "We simply append the text prompt after the visual prompt as input to the LLM." (Section 4.1)
- Inference: Mapped images to 2D (x, y) and prompts/outputs to 1D (t) per glossary based on the quoted instruction-following generation description.

### Task: Image captioning
- "We finetune BLIP-2 models for the image captioning task, which asks the model to generate a text description for the image's visual content." (Section 4.2)
- Inference: Mapped images to 2D (x, y) and captions to 1D (t) per glossary based on the quoted captioning description.

### Task: Visual question answering (VQA)
- "We perform quantitative evaluation on the zero-shot visual question answering task." (Section 4.1)
- "the LLM receives Q-Former's output and the question as input, and is asked to generate the answer." (Section 4.3)
- Inference: Mapped images to 2D (x, y), questions/answers to 1D (t) per glossary based on the quoted VQA input/output description.

### Task: Image-to-text retrieval
- "We then evaluate the model for both image-to-text retrieval and text-to-image retrieval on COCO and Flickr30K." (Section 4.4)
- Inference: Mapped images to 2D (x, y) and retrieved text to 1D (t) per glossary based on the quoted image-to-text retrieval description.

### Task: Text-to-image retrieval
- "We then evaluate the model for both image-to-text retrieval and text-to-image retrieval on COCO and Flickr30K." (Section 4.4)
- Inference: Mapped text to 1D (t) and retrieved images to 2D (x, y) per glossary based on the quoted text-to-image retrieval description.
