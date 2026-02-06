# CIRCLE-ROPE: CONE-LIKE DECOUPLED ROTARY POSI-TIONAL EMBEDDING FOR LARGE VISION-LANGUAGE MODELS (Year not specified)
Source: Circle-RoPE- Cone-like Decoupled Rotary Positional Embedding for Large Vision-Language Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Multimodal understanding and reasoning (MMMU) (inferred) | text tokens; image tokens | 1D (t); 2D (x, y) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Multimodal understanding (MMMU-Pro) (inferred) | text tokens; image tokens | 1D (t); 2D (x, y) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Mathematical reasoning in visual contexts (MathVista) (inferred) | text tokens; image tokens | 1D (t); 2D (x, y) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Vision-language model evaluation (MMStar) (inferred) | text tokens; image tokens | 1D (t); 2D (x, y) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Diagram understanding (AI2D) (inferred) | text tokens; image tokens | 1D (t); 2D (x, y) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Real-world question answering (RealWorldQA) (inferred) | text tokens; image tokens | 1D (t); 2D (x, y) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Infographic visual question answering (InfoVQA) (inferred) | text tokens; image tokens | 1D (t); 2D (x, y) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Chart question answering (ChartQA) (inferred) | text tokens; image tokens | 1D (t); 2D (x, y) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Math-vision evaluation (MathVision) (inferred) | text tokens; image tokens | 1D (t); 2D (x, y) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper evaluates a vision-language model across multiple named benchmarks (e.g., MMMU, MathVista, MMStar, AI2D, InfoVQA, ChartQA, MathVision), but it does not specify task I/O details beyond indicating text and image tokens. It explicitly describes inputs as text tokens with 1D indices and image tokens with 2D (x, y) indices, while outputs, dynamics, attention dynamics, and state dynamics are not specified. Task intents such as multimodal understanding, reasoning, and QA are inferred from benchmark names or titles in the references.

## Evidence
### Task: Multimodal understanding and reasoning (MMMU) (inferred)
- "MMMU <sub>val</sub> [29]" (Section 5.2, Table 2)
- "image token indices are represented separately by width and height coordinates, text tokens use 1D positional index equivalent to standard RoPE." (Section 4.1)
- Inference: Task intent inferred from the benchmark title "Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark" (References)

### Task: Multimodal understanding (MMMU-Pro) (inferred)
- "MMMU-Pro <sub>overall</sub> [30]" (Section 5.2, Table 2)
- "image token indices are represented separately by width and height coordinates, text tokens use 1D positional index equivalent to standard RoPE." (Section 4.1)
- Inference: Task intent inferred from the benchmark title "Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark." (References)

### Task: Mathematical reasoning in visual contexts (MathVista) (inferred)
- "MathVista <sub>mini</sub> [15]" (Section 5.2, Table 2)
- "image token indices are represented separately by width and height coordinates, text tokens use 1D positional index equivalent to standard RoPE." (Section 4.1)
- Inference: Task intent inferred from the benchmark title "Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts." (References)

### Task: Vision-language model evaluation (MMStar) (inferred)
- "MMStar [3]" (Section 5.2, Table 2)
- "image token indices are represented separately by width and height coordinates, text tokens use 1D positional index equivalent to standard RoPE." (Section 4.1)
- Inference: Task intent inferred from the benchmark title "Are we on the right way for evaluating large vision-language models?" (References)

### Task: Diagram understanding (AI2D) (inferred)
- "AI2D [9]" (Section 5.2, Table 2)
- "image token indices are represented separately by width and height coordinates, text tokens use 1D positional index equivalent to standard RoPE." (Section 4.1)
- Inference: Task intent inferred from the benchmark title "A diagram is worth a dozen images." (References)

### Task: Real-world question answering (RealWorldQA) (inferred)
- "RealWorldQA [25]" (Section 5.2, Table 2)
- "image token indices are represented separately by width and height coordinates, text tokens use 1D positional index equivalent to standard RoPE." (Section 4.1)
- Inference: Task intent inferred from the dataset name "RealWorldQA [25]" (Section 5.2, Table 2)

### Task: Infographic visual question answering (InfoVQA) (inferred)
- "InfoVQA [17]" (Section 5.2, Table 2)
- "image token indices are represented separately by width and height coordinates, text tokens use 1D positional index equivalent to standard RoPE." (Section 4.1)
- Inference: Task intent inferred from the benchmark title "Infographicvqa" (References)

### Task: Chart question answering (ChartQA) (inferred)
- "ChartQA_TEST" (Section 5.4, Table 4)
- "image token indices are represented separately by width and height coordinates, text tokens use 1D positional index equivalent to standard RoPE." (Section 4.1)
- Inference: Task intent inferred from the dataset name "ChartQA_TEST" (Section 5.4, Table 4)

### Task: Math-vision evaluation (MathVision) (inferred)
- "MathVision ↑" (Appendix A.1, Table 6)
- "image token indices are represented separately by width and height coordinates, text tokens use 1D positional index equivalent to standard RoPE." (Section 4.1)
- Inference: Task intent inferred from the dataset name "MathVision ↑" (Appendix A.1, Table 6)

---

## CSV Output (required)
Write a CSV file to "/home/jake/Developer/timeline/BIBLIOTHEQUE/01_POS-ENCDR/Circle-RoPE- Cone-like Decoupled Rotary Positional Embedding for Large Vision-Language Models/.TASK-DOMAINS.csv.tmp.0edc4e90096b464f8b257152c540a3e8" with the same rows and columns as the Task
Table. Use the exact header:

task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic

Do not add extra columns, commentary, or blank lines.
