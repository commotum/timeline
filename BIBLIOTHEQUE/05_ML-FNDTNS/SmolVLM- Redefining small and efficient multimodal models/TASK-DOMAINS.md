# SmolVLM: Redefining small and efficient multimodal models (Year not specified in the paper.)
Source: SmolVLM- Redefining small and efficient multimodal models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OCR / character recognition | images containing text; prompt/question text | 2D (x, y) (inferred); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | recognized characters/text responses | 1D (t) (inferred) | Capped (inferred) |
| Image question answering and visual reasoning | single images or multi-image inputs (including documents/charts/tables); question/instruction text | 2D (x, y) (inferred); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text answers/reasoning responses | 1D (t) (inferred) | Capped (inferred) |
| Image/video captioning and visual description generation | images or sampled video frames; prompt text | 2D (x, y) (inferred); 3D (x, y, t) (inferred); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | captions/visual descriptions (text) | 1D (t) (inferred) | Capped (inferred) |
| Video question answering and temporal/narrative comprehension | video frames; question/instruction text | 3D (x, y, t) (inferred); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text answers/comprehension responses | 1D (t) (inferred) | Capped (inferred) |
| Text question answering and reasoning | text prompts/questions (general knowledge, reasoning, math, coding) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text answers/solutions | 1D (t) (inferred) | Capped (inferred) |

## Summary
The OCR supports SmolVLM across OCR/character recognition, image QA/reasoning, image/video captioning, video comprehension (including temporal and narrative tasks), and retained text QA/reasoning. The justified input space spans 1D text, 2D images, and 3D video (frames sampled over time), with outputs described as text. Dynamics are inferred as capped from explicit context limits, and attention/state are inferred as static/direct from the described concatenated visual-text self-attention pipeline that produces direct text outputs.

## Evidence
### Task: OCR / character recognition
- "OCRBench (Liu et al., 2024e)<br>Character Recognition" (Table 1, Section 4.3 Strong Performance at a Tiny Scale)
- "smaller models benefited substantially from positional tokens, achieving notably higher OCR accuracy and improved generalization across tasks." (Section 3.1 Learned Tokens vs. String)
- Inference: `2D (x, y)`, `1D (t)`, `Capped`, `Static`, `Direct`, and text-output dynamics are inferred from "Images are split into subimages, frames are sampled from videos, and then encoded into visual features." and "This combined sequence is passed to the LLM for text output." (Figure 2), plus "Accordingly, we adopt a 16k-token context for SmolVLM and an 8k-token limit for smaller variants." (Section 2.2).

### Task: Image question answering and visual reasoning
- "The visual components comprise document understanding, captioning, and visual question answering (including 2% dedicated to multi-image reasoning), chart understanding, table understanding, and visual reasoning tasks." (Section 4.1 Training Data)
- "This effect is significantly pronounced in multimodal QA, where questions are often repetitive and can be trivially memorized by the model." (Section 3.2 Structured Text Prompts and Media Segmentation)
- Inference: `2D (x, y)` + `1D (t)`, `Capped`, `Static`, `Direct`, and `1D (t)` text output are inferred from the image+text joint-token architecture in Figure 2 and the explicit token limits in Section 2.2 ("16k-token context" / "8k-token limit").

### Task: Image/video captioning and visual description generation
- "Metrics average CIDEr (captioning) and accuracy (visual question answering)." (Figure 3 caption)
- "For video, we sample visual description and captioning from LLaVA-video-178k (Zhang et al., 2024), Video-STAR (Zohar et al., 2024a), Vript (Yang et al., 2024), and ShareGPT4Video (Chen et al., 2023)" (Section 4.1 Training Data)
- Inference: `2D (x, y)`/`3D (x, y, t)` + `1D (t)` inputs, `Capped`, `Static`, `Direct`, and `1D (t)` text outputs are inferred from "frames are sampled from videos" and "passed to the LLM for text output" (Figure 2), together with context-window limits in Section 2.2.

### Task: Video question answering and temporal/narrative comprehension
- "SmolVLM models extend beyond static images, demonstrating robust video comprehension capabilities." (Abstract)
- "For video, we sample visual description and captioning from LLaVA-video-178k (Zhang et al., 2024), Video-STAR (Zohar et al., 2024a), Vript (Yang et al., 2024), and ShareGPT4Video (Chen et al., 2023), temporal understanding from Vista-400k (Ren et al., 2024), and narrative comprehension from MovieChat (Song et al., 2024) and FineVideo (Farré et al., 2024)." (Section 4.1 Training Data)
- Inference: `3D (x, y, t)` + `1D (t)`, `Capped`, `Static`, `Direct`, and `1D (t)` text outputs are inferred from video-frame sampling and text decoding in Figure 2 and capped context limits in Section 2.2.

### Task: Text question answering and reasoning
- "To preserve the model's performance in text-based tasks, we retained a modest amount of general knowledge Q&A and text-based reasoning & logic problems, which incorporate mathematics and coding challenges." (Section 4.1 Training Data)
- "we varied the proportion of CoT data integrated into the Mammoth dataset (Yue et al., 2024b), covering text, image, and video tasks." (Section 3.4 Optimizing Chain-of-Thought Integration for Compact Models)
- Inference: `1D (t)`, `Capped`, `Static`, `Direct`, and `1D (t)` text outputs are inferred from tokenized LM processing and explicit context limits (Section 2.2: "16k-token context" / "8k-token limit").
