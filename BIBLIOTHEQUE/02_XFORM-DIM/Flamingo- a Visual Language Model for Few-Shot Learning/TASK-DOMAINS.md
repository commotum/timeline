# Flamingo: a Visual Language Model for Few-Shot Learning (2022)
Source: Flamingo- a Visual Language Model for Few-Shot Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Object classification | images interleaved with text | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label/choice (finite set) | 0D (inferred) | Fixed (inferred) |
| Scene description | images interleaved with text | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | free-form text description | 1D (t) (inferred) | Capped (inferred) |
| Scene understanding QA | images interleaved with text | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text answer | 1D (t) (inferred) | Capped (inferred) |
| External knowledge QA | images interleaved with text | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text answer | 1D (t) (inferred) | Capped (inferred) |
| Text reading QA | images interleaved with text | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text answer | 1D (t) (inferred) | Capped (inferred) |
| Visual Dialogue | images interleaved with text | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text response | 1D (t) (inferred) | Capped (inferred) |
| Meme classification | images interleaved with text | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label/choice (finite set) | 0D (inferred) | Fixed (inferred) |
| Action classification | videos interleaved with text | 3D (x, y, t); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label/choice (finite set) | 0D (inferred) | Fixed (inferred) |
| Event description | videos interleaved with text | 3D (x, y, t); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | free-form text description | 1D (t) (inferred) | Capped (inferred) |
| Event understanding QA | videos interleaved with text | 3D (x, y, t); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text answer | 1D (t) (inferred) | Capped (inferred) |
| Composite action retrieval | videos interleaved with text | 3D (x, y, t); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label/choice (finite set) | 0D (inferred) | Fixed (inferred) |
| Temporal/Causal QA | videos interleaved with text | 3D (x, y, t); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text answer | 1D (t) (inferred) | Capped (inferred) |
| Multiple-choice QA | videos interleaved with text | 3D (x, y, t); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label/choice (finite set) | 0D (inferred) | Fixed (inferred) |

## Summary
Flamingo is evaluated on image and video benchmarks covering object/action/meme classification, scene/event description, multiple flavors of question answering, visual dialogue, and composite action retrieval. Across these tasks, inputs are interleaved images or videos with text prompts/questions, and outputs are either free-form text or finite-choice labels depending on whether the task is generative. Based on the described interleaved interface and fixed attention masking, the inferred domain structure is 2D images or 3D videos plus 1D text with capped sequence lengths, static attention, and direct autoregressive state.

## Evidence
### Task: Object classification
- "Object classification" (Table 6, Appendix B.1.4)
- "accepts text interleaved with images/videos as input and outputs free-form text." (Section 2)
- "masking the full text-to-image cross-attention matrix, limiting which visual tokens the model sees at each text token." (Section 2.3)
- "up to 32 pairs (or "shots") of images/videos and corresponding texts" (Section 2.3)
- "visually-conditioned autoregressive text generation models" (Introduction)
- "If a task is non-generative it means that we use the VLM to score answers among a given finite set." (Table 6, Appendix B.1.4)
- Inference: In Dimension (2D images + 1D text) and In Dynamics (Capped) are inferred from the interleaved visual-text interface and the capped shots; Attention Dynamic (Static) from the fixed cross-attention masking; State Dynamic (Direct) from the autoregressive modeling; Out Dimension/Out Dynamics (0D, Fixed) from finite-set scoring for non-generative tasks.

### Task: Scene description
- "Scene description" (Table 6, Appendix B.1.4)
- "captioning tasks, which evaluate the ability to describe a scene or an event" (Abstract)
- "accepts text interleaved with images/videos as input and outputs free-form text." (Section 2)
- "masking the full text-to-image cross-attention matrix, limiting which visual tokens the model sees at each text token." (Section 2.3)
- "up to 32 pairs (or "shots") of images/videos and corresponding texts" (Section 2.3)
- "visually-conditioned autoregressive text generation models" (Introduction)
- Inference: In Dimension (2D images + 1D text) and In Dynamics (Capped) are inferred from the interleaved visual-text interface and the capped shots; Attention Dynamic (Static) from the fixed cross-attention masking; State Dynamic (Direct) from the autoregressive modeling; Out Dimension/Out Dynamics (1D, Capped) from free-form text output.

### Task: Scene understanding QA
- "Scene understanding QA" (Table 6, Appendix B.1.4)
- "open-ended tasks such as visual question-answering, where the model is prompted with a question which it has to answer" (Abstract)
- "accepts text interleaved with images/videos as input and outputs free-form text." (Section 2)
- "masking the full text-to-image cross-attention matrix, limiting which visual tokens the model sees at each text token." (Section 2.3)
- "up to 32 pairs (or "shots") of images/videos and corresponding texts" (Section 2.3)
- "visually-conditioned autoregressive text generation models" (Introduction)
- Inference: In Dimension (2D images + 1D text) and In Dynamics (Capped) are inferred from the interleaved visual-text interface and the capped shots; Attention Dynamic (Static) from the fixed cross-attention masking; State Dynamic (Direct) from the autoregressive modeling; Out Dimension/Out Dynamics (1D, Capped) from free-form text output.

### Task: External knowledge QA
- "External knowledge QA" (Table 6, Appendix B.1.4)
- "open-ended tasks such as visual question-answering, where the model is prompted with a question which it has to answer" (Abstract)
- "accepts text interleaved with images/videos as input and outputs free-form text." (Section 2)
- "masking the full text-to-image cross-attention matrix, limiting which visual tokens the model sees at each text token." (Section 2.3)
- "up to 32 pairs (or "shots") of images/videos and corresponding texts" (Section 2.3)
- "visually-conditioned autoregressive text generation models" (Introduction)
- Inference: In Dimension (2D images + 1D text) and In Dynamics (Capped) are inferred from the interleaved visual-text interface and the capped shots; Attention Dynamic (Static) from the fixed cross-attention masking; State Dynamic (Direct) from the autoregressive modeling; Out Dimension/Out Dynamics (1D, Capped) from free-form text output.

### Task: Text reading QA
- "Text reading QA" (Table 6, Appendix B.1.4)
- "TextVQA [100] which specifically assesses OCR capabilities through question-answering;" (Appendix B.1.4)
- "open-ended tasks such as visual question-answering, where the model is prompted with a question which it has to answer" (Abstract)
- "accepts text interleaved with images/videos as input and outputs free-form text." (Section 2)
- "masking the full text-to-image cross-attention matrix, limiting which visual tokens the model sees at each text token." (Section 2.3)
- "up to 32 pairs (or "shots") of images/videos and corresponding texts" (Section 2.3)
- "visually-conditioned autoregressive text generation models" (Introduction)
- Inference: In Dimension (2D images + 1D text) and In Dynamics (Capped) are inferred from the interleaved visual-text interface and the capped shots; Attention Dynamic (Static) from the fixed cross-attention masking; State Dynamic (Direct) from the autoregressive modeling; Out Dimension/Out Dynamics (1D, Capped) from free-form text output.

### Task: Visual Dialogue
- "Visual Dialogue" (Table 6, Appendix B.1.4)
- "VisDial [20], a visual dialogue benchmark;" (Appendix B.1.4)
- "accepts text interleaved with images/videos as input and outputs free-form text." (Section 2)
- "masking the full text-to-image cross-attention matrix, limiting which visual tokens the model sees at each text token." (Section 2.3)
- "up to 32 pairs (or "shots") of images/videos and corresponding texts" (Section 2.3)
- "visually-conditioned autoregressive text generation models" (Introduction)
- Inference: In Dimension (2D images + 1D text) and In Dynamics (Capped) are inferred from the interleaved visual-text interface and the capped shots; Attention Dynamic (Static) from the fixed cross-attention masking; State Dynamic (Direct) from the autoregressive modeling; Out Dimension/Out Dynamics (1D, Capped) from free-form text output.

### Task: Meme classification
- "Meme classification" (Table 6, Appendix B.1.4)
- "HatefulMemes [54] a vision and text classification benchmark;" (Appendix B.1.4)
- "accepts text interleaved with images/videos as input and outputs free-form text." (Section 2)
- "masking the full text-to-image cross-attention matrix, limiting which visual tokens the model sees at each text token." (Section 2.3)
- "up to 32 pairs (or "shots") of images/videos and corresponding texts" (Section 2.3)
- "visually-conditioned autoregressive text generation models" (Introduction)
- "If a task is non-generative it means that we use the VLM to score answers among a given finite set." (Table 6, Appendix B.1.4)
- Inference: In Dimension (2D images + 1D text) and In Dynamics (Capped) are inferred from the interleaved visual-text interface and the capped shots; Attention Dynamic (Static) from the fixed cross-attention masking; State Dynamic (Direct) from the autoregressive modeling; Out Dimension/Out Dynamics (0D, Fixed) from finite-set scoring for non-generative tasks.

### Task: Action classification
- "Action classification" (Table 6, Appendix B.1.4)
- "accepts text interleaved with images/videos as input and outputs free-form text." (Section 2)
- "masking the full text-to-image cross-attention matrix, limiting which visual tokens the model sees at each text token." (Section 2.3)
- "up to 32 pairs (or "shots") of images/videos and corresponding texts" (Section 2.3)
- "visually-conditioned autoregressive text generation models" (Introduction)
- "If a task is non-generative it means that we use the VLM to score answers among a given finite set." (Table 6, Appendix B.1.4)
- Inference: In Dimension (3D videos + 1D text) and In Dynamics (Capped) are inferred from the interleaved visual-text interface and the capped shots; Attention Dynamic (Static) from the fixed cross-attention masking; State Dynamic (Direct) from the autoregressive modeling; Out Dimension/Out Dynamics (0D, Fixed) from finite-set scoring for non-generative tasks.

### Task: Event description
- "Event description" (Table 6, Appendix B.1.4)
- "captioning tasks, which evaluate the ability to describe a scene or an event" (Abstract)
- "accepts text interleaved with images/videos as input and outputs free-form text." (Section 2)
- "masking the full text-to-image cross-attention matrix, limiting which visual tokens the model sees at each text token." (Section 2.3)
- "up to 32 pairs (or "shots") of images/videos and corresponding texts" (Section 2.3)
- "visually-conditioned autoregressive text generation models" (Introduction)
- Inference: In Dimension (3D videos + 1D text) and In Dynamics (Capped) are inferred from the interleaved visual-text interface and the capped shots; Attention Dynamic (Static) from the fixed cross-attention masking; State Dynamic (Direct) from the autoregressive modeling; Out Dimension/Out Dynamics (1D, Capped) from free-form text output.

### Task: Event understanding QA
- "Event understanding QA" (Table 6, Appendix B.1.4)
- "open-ended tasks such as visual question-answering, where the model is prompted with a question which it has to answer" (Abstract)
- "accepts text interleaved with images/videos as input and outputs free-form text." (Section 2)
- "masking the full text-to-image cross-attention matrix, limiting which visual tokens the model sees at each text token." (Section 2.3)
- "up to 32 pairs (or "shots") of images/videos and corresponding texts" (Section 2.3)
- "visually-conditioned autoregressive text generation models" (Introduction)
- Inference: In Dimension (3D videos + 1D text) and In Dynamics (Capped) are inferred from the interleaved visual-text interface and the capped shots; Attention Dynamic (Static) from the fixed cross-attention masking; State Dynamic (Direct) from the autoregressive modeling; Out Dimension/Out Dynamics (1D, Capped) from free-form text output.

### Task: Composite action retrieval
- "Composite action retrieval" (Table 6, Appendix B.1.4)
- "RareAct [73], a benchmark measuring compositionality in action recognition." (Appendix B.1.4)
- "accepts text interleaved with images/videos as input and outputs free-form text." (Section 2)
- "masking the full text-to-image cross-attention matrix, limiting which visual tokens the model sees at each text token." (Section 2.3)
- "up to 32 pairs (or "shots") of images/videos and corresponding texts" (Section 2.3)
- "visually-conditioned autoregressive text generation models" (Introduction)
- "If a task is non-generative it means that we use the VLM to score answers among a given finite set." (Table 6, Appendix B.1.4)
- Inference: In Dimension (3D videos + 1D text) and In Dynamics (Capped) are inferred from the interleaved visual-text interface and the capped shots; Attention Dynamic (Static) from the fixed cross-attention masking; State Dynamic (Direct) from the autoregressive modeling; Out Dimension/Out Dynamics (0D, Fixed) from finite-set scoring for non-generative tasks.

### Task: Temporal/Causal QA
- "Temporal/Causal QA" (Table 6, Appendix B.1.4)
- "NextQA [129] which specially focuses on causality and temporal relation;" (Appendix B.1.4)
- "open-ended tasks such as visual question-answering, where the model is prompted with a question which it has to answer" (Abstract)
- "accepts text interleaved with images/videos as input and outputs free-form text." (Section 2)
- "masking the full text-to-image cross-attention matrix, limiting which visual tokens the model sees at each text token." (Section 2.3)
- "up to 32 pairs (or "shots") of images/videos and corresponding texts" (Section 2.3)
- "visually-conditioned autoregressive text generation models" (Introduction)
- Inference: In Dimension (3D videos + 1D text) and In Dynamics (Capped) are inferred from the interleaved visual-text interface and the capped shots; Attention Dynamic (Static) from the fixed cross-attention masking; State Dynamic (Direct) from the autoregressive modeling; Out Dimension/Out Dynamics (1D, Capped) from free-form text output.

### Task: Multiple-choice QA
- "Multiple-choice QA" (Table 6, Appendix B.1.4)
- "STAR [128], a multiple-choice question answering task;" (Appendix B.1.4)
- "close-ended tasks such as multiple-choice visual question-answering." (Abstract)
- "accepts text interleaved with images/videos as input and outputs free-form text." (Section 2)
- "masking the full text-to-image cross-attention matrix, limiting which visual tokens the model sees at each text token." (Section 2.3)
- "up to 32 pairs (or "shots") of images/videos and corresponding texts" (Section 2.3)
- "visually-conditioned autoregressive text generation models" (Introduction)
- "If a task is non-generative it means that we use the VLM to score answers among a given finite set." (Table 6, Appendix B.1.4)
- Inference: In Dimension (3D videos + 1D text) and In Dynamics (Capped) are inferred from the interleaved visual-text interface and the capped shots; Attention Dynamic (Static) from the fixed cross-attention masking; State Dynamic (Direct) from the autoregressive modeling; Out Dimension/Out Dynamics (0D, Fixed) from finite-set scoring for non-generative tasks.

## CSV Output (required)
