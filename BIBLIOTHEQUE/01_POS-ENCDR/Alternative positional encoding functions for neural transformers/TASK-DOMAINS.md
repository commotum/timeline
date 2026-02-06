# ALTERNATIVE POSITIONAL ENCODING FUNCTIONS FOR NEURAL TRANSFORMERS (2025)
Source: Alternative positional encoding functions for neural transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Machine translation (English-German captions) (inferred) | English caption tokens | 1D (t) | Capped | Static (inferred) | Constructed (inferred) | German caption tokens | 1D (t) | Capped |

## Summary
The paper evaluates a single text-to-text task: English-German caption translation using the Multi30K parallel caption dataset. Inputs and outputs are word-level token sequences with a maximum length of 256 tokens, so the task is 1D and capped. The model is a standard Transformer; attention is static and state is constructed (both inferred).

## Evidence
### Task: Machine translation (English-German captions) (inferred)
- "the Multi30K English–German image–description dataset [Elliott et al., 2016], which provided parallel English–German captions aligned at the sentence level." (Section 3 Experiments)
- "Separate vocabularies were built for the source (English) and target (German) languages" (Section 3 Experiments)
- "All text was lowercased and tokenized at the word level using language–specific tokenizers." (Section 3 Experiments)
- "batches were dynamically padded to the maximum sequence length within each batch, with a maximum allowed length of 256 tokens." (Section 3 Experiments)
- Inference: Task labeled as machine translation and Attention/State marked Static/Constructed because the data are "parallel English–German captions aligned at the sentence level" and a "standard Transformer layer applies content-based self-attention followed by position-wise feed-forward networks." (Sections 3 Experiments; 1 Introduction)

---

## CSV Output (required)
CSV file: `/home/jake/Developer/timeline/BIBLIOTHEQUE/01_POS-ENCDR/Alternative positional encoding functions for neural transformers/.TASK-DOMAINS.csv.tmp.59c01bd5989b4288af94e6ca4d6d8363`
