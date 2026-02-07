# HiRoPE: Length Extrapolation for Code Models Using Hierarchical Position (Not specified in the paper.)
Source: HiRoPE- Length Extrapolation for Code Models Using Hierarchical Rotary Position Embedding.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language modeling (long code) | code tokens (long code sequences) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Not specified in the paper. | code tokens (next-token predictions) | 1D (t) (inferred) | Capped (inferred) |
| Language modeling (long natural language) | text tokens (long natural language sequences) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Not specified in the paper. | text tokens (next-token predictions) | 1D (t) (inferred) | Capped (inferred) |
| Code symbol understanding | code tokens (long code file/context) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Not specified in the paper. | function and class names (tokens/strings) | 1D (t) (inferred) | Capped (inferred) |
| Long code completion (next-line generation) | code tokens (long code context) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Not specified in the paper. | code tokens (next line of code) | 1D (t) (inferred) | Capped (inferred) |

## Summary
Across four evaluated tasks, the paper covers long-sequence language modeling for code and natural language, plus code symbol extraction and long code completion. Inputs and outputs are token sequences, so the task coverage is 1D (t) with capped lengths implied by stated context-length limits (inferred). Attention is treated as static based on fixed self-attention with RoPE (inferred), while state dynamics are not specified.

## Evidence
### Task: Language modeling (long code)
- "language modeling capability of HiRoPE on long code sequences" (Section 4 Experiment Setup)
- "We evaluate Hi-RoPE's language modeling ability on CodeParrot-valid dataset." (Section 5.1 Long Code Language Modeling)
- Inference: In/Out Dimension and Dynamics are 1D (t) and Capped because the paper ties inputs to token context length ("LLMs are typically pre-trained with a context length ranging from 2k to 16k tokens"). Attention is Static because fixed self-attention is used ("RoPE is applied on both query and key embeddings for computing attention scores"). (Section 1 Introduction; Section 2.1 Rotary Position Embedding in Transformer)

### Task: Language modeling (long natural language)
- "language modeling capability of HiRoPE on long natural language sequences" (Section 4 Experiment Setup)
- "we also evaluate its effects on long natural language texts." (Section 5.2 Long Text Language Modeling)
- Inference: In/Out Dimension and Dynamics are 1D (t) and Capped because the paper ties inputs to token context length ("LLMs are typically pre-trained with a context length ranging from 2k to 16k tokens"). Attention is Static because fixed self-attention is used ("RoPE is applied on both query and key embeddings for computing attention scores"). (Section 1 Introduction; Section 2.1 Rotary Position Embedding in Transformer)

### Task: Code symbol understanding
- "we design a new evaluation task on real code projects" (Section 4 Experiment Setup)
- "the model is required to output all the function names and class names defined in it." (Section 4.4 Details of Code Symbol Understanding task)
- Inference: In/Out Dimension and Dynamics are 1D (t) and Capped because the paper ties inputs to token context length ("LLMs are typically pre-trained with a context length ranging from 2k to 16k tokens"). Attention is Static because fixed self-attention is used ("RoPE is applied on both query and key embeddings for computing attention scores"). (Section 1 Introduction; Section 2.1 Rotary Position Embedding in Transformer)

### Task: Long code completion (next-line generation)
- "How does HiRoPE perform on existing benchmarks for long code completion?" (Section 4 Experiment Setup)
- "the model is required to generate the complete next line of code." (Section 5.4 Long Code Completion)
- Inference: In/Out Dimension and Dynamics are 1D (t) and Capped because the paper ties inputs to token context length ("LLMs are typically pre-trained with a context length ranging from 2k to 16k tokens"). Attention is Static because fixed self-attention is used ("RoPE is applied on both query and key embeddings for computing attention scores"). (Section 1 Introduction; Section 2.1 Rotary Position Embedding in Transformer)

---

## CSV Output (required)
task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic
Language modeling (long code),code tokens (long code sequences),1D (t) (inferred),Capped (inferred),Static (inferred),Not specified in the paper.,code tokens (next-token predictions),1D (t) (inferred),Capped (inferred)
Language modeling (long natural language),text tokens (long natural language sequences),1D (t) (inferred),Capped (inferred),Static (inferred),Not specified in the paper.,text tokens (next-token predictions),1D (t) (inferred),Capped (inferred)
Code symbol understanding,code tokens (long code file/context),1D (t) (inferred),Capped (inferred),Static (inferred),Not specified in the paper.,function and class names (tokens/strings),1D (t) (inferred),Capped (inferred)
Long code completion (next-line generation),code tokens (long code context),1D (t) (inferred),Capped (inferred),Static (inferred),Not specified in the paper.,code tokens (next line of code),1D (t) (inferred),Capped (inferred)
