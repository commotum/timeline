# IMPROVING MOE COMPUTE EFFICIENCY BY COMPOSING WEIGHT AND DATA SPARSITY (Not specified in the paper.)
Source: Improving MoE Compute Efficiency by Composing Weight and Data Sparsity.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pointing / grounding | text and image patches | 1D (t) (inferred); 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| OCR | text and image patches | 1D (t) (inferred); 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Counting | text and image patches | 1D (t) (inferred); 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| General vision-language QA | text and image patches | 1D (t) (inferred); 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Not specified in the paper. | text answer (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Segmentation (prompted) | text and image patches | 1D (t) (inferred); 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper evaluates a multimodal vision-language MoE across pointing/grounding, OCR, counting, and general vision-language QA benchmarks, and it also analyzes a targeted segmentation prompt as a task context. Inputs are described as text and image patches, implying 1D (t) and 2D (x, y) inputs with capped sequence lengths (inferred from the stated sequence lengths). Routing behavior indicates dynamic attention to tokens based on task context (inferred), while output formats and state dynamics are largely not specified except for a QA prompt requesting a single-word text answer.

## Evidence
### Task: Pointing / grounding
- "| Pointing | Aerial Grounding      | 80.7                      | 82.2                        | 73.1      | _                   |" (Table 1)
- "across pointing, OCR, counting, and general vision-language tasks" (Table 1)
- "Original input (text and image patches)." (Figure 5)
- Inference: In Dimension set to 1D (t) and 2D (x, y) because the input is described as "text and image patches" (Figure 5). In Dynamics set to Capped from "sequence length 2048" and "sequence length 8192" in the training details (Section 5.1). Attention Dynamic set to Dynamic from "routing low-information tokens to null experts while preserving compute for tokens that need it" and "task-dependent: the same image receives different compute maps under different prompts" (Section 6).

### Task: OCR
- "| OCR      | ChartQA [23]          | 79.0                      | 80.3                        | 75.1      | 86.6                |" (Table 1)
- "The improvements are particularly pronounced in OCR and counting tasks." (Section 5.4)
- "Original input (text and image patches)." (Figure 5)
- Inference: In Dimension set to 1D (t) and 2D (x, y) because the input is described as "text and image patches" (Figure 5). In Dynamics set to Capped from "sequence length 2048" and "sequence length 8192" in the training details (Section 5.1). Attention Dynamic set to Dynamic from "routing low-information tokens to null experts while preserving compute for tokens that need it" and "task-dependent: the same image receives different compute maps under different prompts" (Section 6).

### Task: Counting
- "| Counting | Aerial Counting       | 53.0                      | 57.0                        | 52.0      |                     |" (Table 1)
- "The improvements are particularly pronounced in OCR and counting tasks." (Section 5.4)
- "Original input (text and image patches)." (Figure 5)
- Inference: In Dimension set to 1D (t) and 2D (x, y) because the input is described as "text and image patches" (Figure 5). In Dynamics set to Capped from "sequence length 2048" and "sequence length 8192" in the training details (Section 5.1). Attention Dynamic set to Dynamic from "routing low-information tokens to null experts while preserving compute for tokens that need it" and "task-dependent: the same image receives different compute maps under different prompts" (Section 6).

### Task: General vision-language QA
- "| General  | VSR (Zero-Shot) [28]  | 79.6                      | 80.6                        | 78.6      | _                   |" (Table 1)
- "Compute overlay under a general QA prompt—high compute distributed broadly." (Figure 6)
- "underspecified prompt (i.e., \"Answer with single word.\")" (Section 6.2)
- "Original input (text and image patches)." (Figure 5)
- Inference: In Dimension set to 1D (t) and 2D (x, y) because the input is described as "text and image patches" (Figure 5). In Dynamics set to Capped from "sequence length 2048" and "sequence length 8192" in the training details (Section 5.1). Attention Dynamic set to Dynamic from "routing low-information tokens to null experts while preserving compute for tokens that need it" and "task-dependent: the same image receives different compute maps under different prompts" (Section 6). Output and Out Dimension inferred as text sequence from the prompt instruction "Answer with single word." (Section 6.2).

### Task: Segmentation (prompted)
- "Compute overlay under a targeted segmentation prompt—reduced overall compute, concentrated on task-relevant regions." (Figure 6)
- "Under segmentation, it concentrates compute on task-relevant regions and routes most patches to null experts." (Section 6.2)
- "Original input (text and image patches)." (Figure 5)
- Inference: In Dimension set to 1D (t) and 2D (x, y) because the input is described as "text and image patches" (Figure 5). In Dynamics set to Capped from "sequence length 2048" and "sequence length 8192" in the training details (Section 5.1). Attention Dynamic set to Dynamic from "routing low-information tokens to null experts while preserving compute for tokens that need it" and "task-dependent: the same image receives different compute maps under different prompts" (Section 6).
