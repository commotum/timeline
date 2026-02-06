# ConceptMoE: Adaptive Token-to-Concept Compression for Implicit Compute Allocation (2026)
Source: ConceptMoE- Adaptive Token-to-Concept Compression for Implicit Compute Allocation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text reasoning | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| mathematics | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| code generation | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| knowledge retrieval | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| needle-in-haystack | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| long context summarization | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| long context understanding | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| visual localization | visual tokens + text tokens (inferred) | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| visual reasoning | visual tokens + text tokens (inferred) | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| hallucination detection | visual tokens + text tokens (inferred) | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| visual question answering | visual tokens + text tokens (inferred) | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| chart extraction | visual tokens + text tokens (inferred) | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
ConceptMoE is evaluated on a wide set of text-centric tasks (reasoning, math, code generation, knowledge retrieval, needle-in-haystack, and long-context summarization/understanding) and multimodal vision-language tasks (visual localization, visual reasoning, hallucination detection, visual question answering, and chart extraction). The paper describes token-based processing for text and, in the VLM setting, both visual and textual tokens, corresponding to 1D (t) and 2D (x, y) input dimensions with capped sequence lengths. Based on the adaptive chunking and concept-representation architecture, attention is classified as dynamic and state as constructed across tasks, with outputs inferred as token sequences in 1D (t) with capped dynamics.

## Evidence
### Task: text reasoning
- "The evaluation suite covers text reasoning, mathematics, code generation, knowledge retrieval" (Section 4, Evaluation Benchmarks)
- "Large language models (LLMs) process text uniformly at the token level." (Section 1, Introduction)
- "We introduce ConceptMoE, which dynamically merges semantically similar tokens into concept representations" (Abstract)
- "The input consists of 1024 tokens, which are compressed to 701 tokens" (Appendix B)
- "During token decoding, multiple tokens share the same concept" (Section 3.4, Joint decoding)
- Inference: Classified inputs/outputs as text token sequences with 1D (t) and capped dynamics, and attention/state as Dynamic/Constructed based on the quoted token-level processing, fixed-length example, token decoding, and dynamic concept merging.

### Task: mathematics
- "The evaluation suite covers text reasoning, mathematics, code generation, knowledge retrieval" (Section 4, Evaluation Benchmarks)
- "Large language models (LLMs) process text uniformly at the token level." (Section 1, Introduction)
- "We introduce ConceptMoE, which dynamically merges semantically similar tokens into concept representations" (Abstract)
- "The input consists of 1024 tokens, which are compressed to 701 tokens" (Appendix B)
- "During token decoding, multiple tokens share the same concept" (Section 3.4, Joint decoding)
- Inference: Classified inputs/outputs as text token sequences with 1D (t) and capped dynamics, and attention/state as Dynamic/Constructed based on the quoted token-level processing, fixed-length example, token decoding, and dynamic concept merging.

### Task: code generation
- "The evaluation suite covers text reasoning, mathematics, code generation, knowledge retrieval" (Section 4, Evaluation Benchmarks)
- "Large language models (LLMs) process text uniformly at the token level." (Section 1, Introduction)
- "We introduce ConceptMoE, which dynamically merges semantically similar tokens into concept representations" (Abstract)
- "The input consists of 1024 tokens, which are compressed to 701 tokens" (Appendix B)
- "During token decoding, multiple tokens share the same concept" (Section 3.4, Joint decoding)
- Inference: Classified inputs/outputs as text token sequences with 1D (t) and capped dynamics, and attention/state as Dynamic/Constructed based on the quoted token-level processing, fixed-length example, token decoding, and dynamic concept merging.

### Task: knowledge retrieval
- "The evaluation suite covers text reasoning, mathematics, code generation, knowledge retrieval" (Section 4, Evaluation Benchmarks)
- "Large language models (LLMs) process text uniformly at the token level." (Section 1, Introduction)
- "We introduce ConceptMoE, which dynamically merges semantically similar tokens into concept representations" (Abstract)
- "The input consists of 1024 tokens, which are compressed to 701 tokens" (Appendix B)
- "During token decoding, multiple tokens share the same concept" (Section 3.4, Joint decoding)
- Inference: Classified inputs/outputs as text token sequences with 1D (t) and capped dynamics, and attention/state as Dynamic/Constructed based on the quoted token-level processing, fixed-length example, token decoding, and dynamic concept merging.

### Task: needle-in-haystack
- "needle-in-haystack, long context summarization and understanding" (Section 4, Evaluation Benchmarks)
- "Large language models (LLMs) process text uniformly at the token level." (Section 1, Introduction)
- "We introduce ConceptMoE, which dynamically merges semantically similar tokens into concept representations" (Abstract)
- "The input consists of 1024 tokens, which are compressed to 701 tokens" (Appendix B)
- "During token decoding, multiple tokens share the same concept" (Section 3.4, Joint decoding)
- Inference: Classified inputs/outputs as text token sequences with 1D (t) and capped dynamics, and attention/state as Dynamic/Constructed based on the quoted token-level processing, fixed-length example, token decoding, and dynamic concept merging.

### Task: long context summarization
- "needle-in-haystack, long context summarization and understanding" (Section 4, Evaluation Benchmarks)
- "Large language models (LLMs) process text uniformly at the token level." (Section 1, Introduction)
- "We introduce ConceptMoE, which dynamically merges semantically similar tokens into concept representations" (Abstract)
- "The input consists of 1024 tokens, which are compressed to 701 tokens" (Appendix B)
- "During token decoding, multiple tokens share the same concept" (Section 3.4, Joint decoding)
- Inference: Classified inputs/outputs as text token sequences with 1D (t) and capped dynamics, and attention/state as Dynamic/Constructed based on the quoted token-level processing, fixed-length example, token decoding, and dynamic concept merging.

### Task: long context understanding
- "needle-in-haystack, long context summarization and understanding" (Section 4, Evaluation Benchmarks)
- "Large language models (LLMs) process text uniformly at the token level." (Section 1, Introduction)
- "We introduce ConceptMoE, which dynamically merges semantically similar tokens into concept representations" (Abstract)
- "The input consists of 1024 tokens, which are compressed to 701 tokens" (Appendix B)
- "During token decoding, multiple tokens share the same concept" (Section 3.4, Joint decoding)
- Inference: Classified inputs/outputs as text token sequences with 1D (t) and capped dynamics, and attention/state as Dynamic/Constructed based on the quoted token-level processing, fixed-length example, token decoding, and dynamic concept merging.

### Task: visual localization
- "multimodal tasks including visual localization, visual reasoning, hallucination detection, visual question answering, and chart extraction." (Section 4, Evaluation Benchmarks)
- "we apply compression to both visual and textual tokens." (Section 4.2, Train a vision-language model)
- "Large language models (LLMs) process text uniformly at the token level." (Section 1, Introduction)
- "We introduce ConceptMoE, which dynamically merges semantically similar tokens into concept representations" (Abstract)
- "The input consists of 1024 tokens, which are compressed to 701 tokens" (Appendix B)
- "During token decoding, multiple tokens share the same concept" (Section 3.4, Joint decoding)
- Inference: Classified inputs as visual + text tokens with 2D (x, y) and 1D (t) dimensions and capped dynamics, outputs as text tokens with 1D (t) and capped dynamics, and attention/state as Dynamic/Constructed based on the quoted visual/text token processing, token-level modeling, fixed-length example, token decoding, and dynamic concept merging.

### Task: visual reasoning
- "multimodal tasks including visual localization, visual reasoning, hallucination detection, visual question answering, and chart extraction." (Section 4, Evaluation Benchmarks)
- "we apply compression to both visual and textual tokens." (Section 4.2, Train a vision-language model)
- "Large language models (LLMs) process text uniformly at the token level." (Section 1, Introduction)
- "We introduce ConceptMoE, which dynamically merges semantically similar tokens into concept representations" (Abstract)
- "The input consists of 1024 tokens, which are compressed to 701 tokens" (Appendix B)
- "During token decoding, multiple tokens share the same concept" (Section 3.4, Joint decoding)
- Inference: Classified inputs as visual + text tokens with 2D (x, y) and 1D (t) dimensions and capped dynamics, outputs as text tokens with 1D (t) and capped dynamics, and attention/state as Dynamic/Constructed based on the quoted visual/text token processing, token-level modeling, fixed-length example, token decoding, and dynamic concept merging.

### Task: hallucination detection
- "multimodal tasks including visual localization, visual reasoning, hallucination detection, visual question answering, and chart extraction." (Section 4, Evaluation Benchmarks)
- "we apply compression to both visual and textual tokens." (Section 4.2, Train a vision-language model)
- "Large language models (LLMs) process text uniformly at the token level." (Section 1, Introduction)
- "We introduce ConceptMoE, which dynamically merges semantically similar tokens into concept representations" (Abstract)
- "The input consists of 1024 tokens, which are compressed to 701 tokens" (Appendix B)
- "During token decoding, multiple tokens share the same concept" (Section 3.4, Joint decoding)
- Inference: Classified inputs as visual + text tokens with 2D (x, y) and 1D (t) dimensions and capped dynamics, outputs as text tokens with 1D (t) and capped dynamics, and attention/state as Dynamic/Constructed based on the quoted visual/text token processing, token-level modeling, fixed-length example, token decoding, and dynamic concept merging.

### Task: visual question answering
- "multimodal tasks including visual localization, visual reasoning, hallucination detection, visual question answering, and chart extraction." (Section 4, Evaluation Benchmarks)
- "we apply compression to both visual and textual tokens." (Section 4.2, Train a vision-language model)
- "Large language models (LLMs) process text uniformly at the token level." (Section 1, Introduction)
- "We introduce ConceptMoE, which dynamically merges semantically similar tokens into concept representations" (Abstract)
- "The input consists of 1024 tokens, which are compressed to 701 tokens" (Appendix B)
- "During token decoding, multiple tokens share the same concept" (Section 3.4, Joint decoding)
- Inference: Classified inputs as visual + text tokens with 2D (x, y) and 1D (t) dimensions and capped dynamics, outputs as text tokens with 1D (t) and capped dynamics, and attention/state as Dynamic/Constructed based on the quoted visual/text token processing, token-level modeling, fixed-length example, token decoding, and dynamic concept merging.

### Task: chart extraction
- "multimodal tasks including visual localization, visual reasoning, hallucination detection, visual question answering, and chart extraction." (Section 4, Evaluation Benchmarks)
- "we apply compression to both visual and textual tokens." (Section 4.2, Train a vision-language model)
- "Large language models (LLMs) process text uniformly at the token level." (Section 1, Introduction)
- "We introduce ConceptMoE, which dynamically merges semantically similar tokens into concept representations" (Abstract)
- "The input consists of 1024 tokens, which are compressed to 701 tokens" (Appendix B)
- "During token decoding, multiple tokens share the same concept" (Section 3.4, Joint decoding)
- Inference: Classified inputs as visual + text tokens with 2D (x, y) and 1D (t) dimensions and capped dynamics, outputs as text tokens with 1D (t) and capped dynamics, and attention/state as Dynamic/Constructed based on the quoted visual/text token processing, token-level modeling, fixed-length example, token decoding, and dynamic concept merging.
