# Visual Instruction Tuning (2023)
Source: Visual Instruction Tuning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Multimodal visual question answering (instruction-following chat) | Image plus language instructions/questions in dialogue turns | 2D (x, y); 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | Natural-language answers/responses | 1D (t) (inferred) | Open (inferred) |
| Image description generation | Image plus a text instruction to describe the image | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Textual image description/caption | 1D (t) (inferred) | Capped (inferred) |
| Multimodal science question answering with rationale and answer selection | Science question with context in natural language or image | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Reasoning text plus selected answer choice | 1D (t); 0D (inferred) | Capped (inferred) |

## Summary
The paper covers multimodal instruction-following over image and text, primarily through chat-style question answering, image description, and science question answering with explanations. Inputs span 2D images and 1D language sequences; outputs are mostly 1D language, with ScienceQA additionally requiring a discrete answer choice (0D). The chatbot setting is supported as Open dynamics because the paper explicitly uses multi-turn conversations, while single-turn description and benchmark QA are Capped. Attention and state are best supported as Static and Direct from the autoregressive instruction-tuning setup (inferred).

## Evidence
### Task: Multimodal visual question answering (instruction-following chat)
- "Multimodal Chatbot. We develop a Chatbot by fine-tuning on the 158K language-image instruction-following data in Section 3." (Section 4.2, Stage 2: Fine-tuning End-to-End)
- "Among the three types of responses, conversation is multi-turn while the other two are single-turn." (Section 4.2, Stage 2: Fine-tuning End-to-End)
- "We design a conversation between the assistant and a person asking questions about this photo." (Section 3)
- "Complex reasoning. The above two types focus on the visual content itself, based on which we further create in-depth reasoning questions." (Section 3)
- Inference: In Dimension is labeled "2D (x, y); 1D (t)" because the model takes images and language instructions; Open dynamics is inferred from "conversation is multi-turn" and variable-turn training; Attention is Static and State is Direct (reactive autoregressive prediction) from "we perform instruction-tuning ... using its original auto-regressive training objective" (Section 4.2).

### Task: Image description generation
- "Detailed description. To include a rich and comprehensive description for an image, we create a list of questions with such an intent." (Section 3)
- "for an image X_v, a question X_q is randomly sampled, which is a language instruction to request the assistant to describe the image briefly." (Section 4.2, Stage 1: Pre-training for Feature Alignment)
- "The ground-truth prediction answer X_a is the original caption." (Section 4.2, Stage 1: Pre-training for Feature Alignment)
- Inference: In Dimension is "2D (x, y); 1D (t)" and Out Dimension is "1D (t)" from image-plus-text input and text description output; Capped dynamics is inferred because this is organized as single-turn sequence prediction in training (Section 4.2); Attention and State are inferred as Static and Direct from the same autoregressive setup.

### Task: Multimodal science question answering with rationale and answer selection
- "Science QA. We study our method on the ScienceQA benchmark [34], the first large-scale multimodal science question dataset" (Section 4.2, Stage 2: Fine-tuning End-to-End)
- "Each question is provided a context in the form of natural language or an image." (Section 4.2, Stage 2: Fine-tuning End-to-End)
- "The assistant provides the reasoning process in natural language and selects the answer among multiple choices." (Section 4.2, Stage 2: Fine-tuning End-to-End)
- Inference: In Dimension is "1D (t); 2D (x, y)" from language/image context; Out Dimension is "1D (t); 0D" because outputs include free-form reasoning plus a selected option; Capped dynamics is inferred from benchmark single-turn QA format; Attention/State are inferred as Static/Direct from autoregressive instruction tuning.
