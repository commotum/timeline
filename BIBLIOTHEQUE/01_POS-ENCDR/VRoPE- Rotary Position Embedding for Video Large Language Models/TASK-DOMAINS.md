# VRoPE: Rotary Position Embedding for Video Large Language Models (Year not specified in the paper)
Source: VRoPE- Rotary Position Embedding for Video Large Language Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Video understanding | Video frames/tokens and text tokens | 3D (x, y, t); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Text responses (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Temporal reasoning | Video frames/tokens and text tokens | 3D (x, y, t); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Text responses (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Long video retrieval | Long video frame sequences and text query tokens (inferred) | 3D (x, y, t); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Retrieved needle frame position (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper covers three intent-level task areas: video understanding, temporal reasoning, and long video retrieval. The described pipeline is multimodal (video plus text), with explicit video coordinates and text-token concatenation, so the supported input structure is 3D (x, y, t) with 1D (t) text (inferred). Based on the reported frame-count regimes (16/32 in setup and 256-1216 in long-retrieval evaluation), task dynamics are best classified as Capped (inferred), and the transformer self-attention pipeline is treated as Static attention with Direct state (inferred).

## Evidence
### Task: Video understanding
- "We evaluated VRoPE across diverse video benchmarks, covering *general video understanding* (Video-MME (Fu et al., 2024)), *video temporal understanding* (MVBench (Li et al., 2024b), TempCompass (Liu et al., 2024c)), *long video understanding* (MLVU (Zhou et al., 2024), LongVideoBench (Wu et al., 2025), EgoSchema (Mangalam et al., 2024)), and *long video retrieval* (Video-NIAH (Zhao et al., 2024)) to validate its effectiveness. The evaluation is conducted using the official code provided by each benchmark." (Section 5.1 Experimental Setup)
- "In Video-LLMs, video frames are typically processed by vision encoders (e.g., ViTs (Alexey, 2020) or CNNs (He et al., 2016)) and transformed into a sequence of visual tokens. These visual tokens are then concatenated with text tokens and fed into an LLM backbone." (Section 3.2 RoPE for Video-LLMs)
- "This deficiency leads to localization errors and subsequent incorrect responses" (Section C Visualization Analysis)
- Inference: Input dimension is marked as "3D (x, y, t); 1D (t)" from explicit spatiotemporal video-token indexing plus concatenated text tokens; dynamics are marked "Capped" from explicit frame-count settings (16 frames in setup, 32 in larger setup), with long-range evaluation still using finite tested ranges; attention is marked "Static" because the model consumes a predefined token sequence with self-attention; state is marked "Direct" because no explicit persistent constructed state is described; output is marked as text responses from the LLM behavior and the paper’s mention of "incorrect responses."

### Task: Temporal reasoning
- "Extensive experiments on different models demonstrate that VRoPE consistently outperforms previous RoPE variants, achieving significant improvements in video understanding, temporal reasoning, and retrieval tasks." (Abstract)
- "We evaluated VRoPE across diverse video benchmarks, covering *general video understanding* (Video-MME (Fu et al., 2024)), *video temporal understanding* (MVBench (Li et al., 2024b), TempCompass (Liu et al., 2024c)), *long video understanding* (MLVU (Zhou et al., 2024), LongVideoBench (Wu et al., 2025), EgoSchema (Mangalam et al., 2024)), and *long video retrieval* (Video-NIAH (Zhao et al., 2024)) to validate its effectiveness. The evaluation is conducted using the official code provided by each benchmark." (Section 5.1 Experimental Setup)
- "we conduct additional evaluations focusing on event-based tasks involving complex temporal dependencies." (Section B.1 Results on EventBench)
- Inference: The same multimodal token interface and spatiotemporal indexing support the "3D (x, y, t); 1D (t)" input-dimension assignment; "Capped" dynamics follow the finite frame-count settings and bounded benchmark evaluations; output is inferred as 1D text responses from the LLM-based video-language setup; attention/state are inferred as "Static" and "Direct" under the same transformer processing description.

### Task: Long video retrieval
- "We evaluated VRoPE across diverse video benchmarks, covering *general video understanding* (Video-MME (Fu et al., 2024)), *video temporal understanding* (MVBench (Li et al., 2024b), TempCompass (Liu et al., 2024c)), *long video understanding* (MLVU (Zhou et al., 2024), LongVideoBench (Wu et al., 2025), EgoSchema (Mangalam et al., 2024)), and *long video retrieval* (Video-NIAH (Zhao et al., 2024)) to validate its effectiveness. The evaluation is conducted using the official code provided by each benchmark." (Section 5.1 Experimental Setup)
- "We compare our method with RoPE (Su et al., 2024) and RoPE-3D (Wang et al., 2024) on the long video retrieval task to evaluate the model's generalization ability with longer video inputs." (Section 5.3 Results on Long Video Retrieval)
- "Table 4: Average retrieval accuracy across different input frame length intervals on Video-NIAH (Zhao et al., 2024)." (Table 4)
- "Our VRoPE consistently achieves high accuracy across varying background lengths and needle depths, showing strong retrieval capability in long videos." (Figure 4 caption)
- Inference: Output is inferred as selecting/localizing the target needle frame position in time, so output dimension is marked "1D (t)"; input and output dynamics are marked "Capped" because the task is evaluated over bounded frame-count intervals; attention/state are inferred as "Static" and "Direct" given the same self-attention Video-LLM pipeline without described runtime observation-selection or explicit persistent constructed memory.
