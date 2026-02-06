# CoPE: A Lightweight Complex Positional Encoding (Not specified in the paper.)
Source: CoPE- A Lightweight Complex Positional Encoding.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Paraphrase classification (MRPC) (inferred) | sentence pairs (text) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | semantic equivalence label (inferred) | 0D (inferred) | Fixed (inferred) |
| Sentiment classification (SST-2) (inferred) | single movie review sentences | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | sentiment label (inferred) | 0D (inferred) | Fixed (inferred) |
| Answer sentence classification (QNLI) (inferred) | question paired with a sentence from a passage | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | answer-containing label (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates three GLUE text classification tasks: MRPC (sentence-pair paraphrase), SST-2 (single-sentence sentiment), and QNLI (question-sentence answer matching). Inputs are sentence-level token sequences, so the task dimension is 1D and the interface is capped by the stated max positions of 512, while outputs are single decision labels (0D, fixed). The model is a multi-layer Transformer with phase-aware attention in the first layer, so attention is static and state is constructed (both inferred from the fixed-window Transformer setup).

## Evidence
### Task: Paraphrase classification (MRPC) (inferred)
- "MRPC: The Microsoft Research Paraphrase Corpus consists of sentence pairs and evaluates whether two sentences are semantically equivalent." (Section 4.1 Experimental Setup)
- "We use transformer model with 6 layers, 8 heads, 256-dimensional embeddings, 256-dimensional attention with max positions 512." (Section 4.1 Experimental Setup)
- "We introduce phase-aware attention in the first layer of the transformer model to capture position-dependent patterns, followed by standard attention layers for higher-levels." (Abstract)
- Inference: Task labeled as classification with 1D (t) inputs because MRPC is described as "sentence pairs" evaluated for semantic equivalence; dynamics are capped due to "max positions 512"; attention is static and state is constructed because the model is a fixed-window multi-layer Transformer; output is a single equivalence decision label (0D, fixed). (Section 4.1 Experimental Setup; Abstract)

### Task: Sentiment classification (SST-2) (inferred)
- "SST2: The Stanford Sentiment Treebank (binary classification version) uses single movie review sentences to assess sentiment." (Section 4.1 Experimental Setup)
- "We use transformer model with 6 layers, 8 heads, 256-dimensional embeddings, 256-dimensional attention with max positions 512." (Section 4.1 Experimental Setup)
- "We introduce phase-aware attention in the first layer of the transformer model to capture position-dependent patterns, followed by standard attention layers for higher-levels." (Abstract)
- Inference: Task labeled as sentiment classification with 1D (t) inputs because SST-2 uses "single movie review sentences"; dynamics are capped due to "max positions 512"; attention is static and state is constructed because the model is a fixed-window multi-layer Transformer; output is a single sentiment label (0D, fixed). (Section 4.1 Experimental Setup; Abstract)

### Task: Answer sentence classification (QNLI) (inferred)
- "QNLI: The Question-answering Natural Language Inference dataset is a question paired with a sentence from a passage. The task is to determine if the sentence contains the answer to the question." (Section 4.1 Experimental Setup)
- "We use transformer model with 6 layers, 8 heads, 256-dimensional embeddings, 256-dimensional attention with max positions 512." (Section 4.1 Experimental Setup)
- "We introduce phase-aware attention in the first layer of the transformer model to capture position-dependent patterns, followed by standard attention layers for higher-levels." (Abstract)
- Inference: Task labeled as answer sentence classification with 1D (t) inputs because QNLI is described as a question paired with a sentence and a determination of whether the sentence contains the answer; dynamics are capped due to "max positions 512"; attention is static and state is constructed because the model is a fixed-window multi-layer Transformer; output is a single answer-containing label (0D, fixed). (Section 4.1 Experimental Setup; Abstract)

---

## CSV Output (required)
CSV file: `/home/jake/Developer/timeline/BIBLIOTHEQUE/01_POS-ENCDR/CoPE- A Lightweight Complex Positional Encoding/.TASK-DOMAINS.csv.tmp.31970a989f054319b646187f3221259a`
