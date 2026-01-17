## 1. Basic Metadata

- Title: "CoPE: A Lightweight Complex Positional Encoding" (Title)
- Authors: "Avinash Amballa" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper introduces CoPE, "a lightweight Complex Positional Encoding" that "leverages complex-valued encoding to encode both content and positional information" and adds "phase-aware attention in the first layer of the transformer model." (Abstract)

---

## 3. Tasks Evaluated

Paper-level task listing: "We experiment with several datasets from GLUE, i.e. MRPC Dolan and Brockett [2005], SST-2 Socher et al. [2013], QNLI for training tasks." (Section 4.1 Experimental Setup)

### Task: MRPC

- Task name: MRPC (paraphrase detection/classification)
- Task type: Classification
- Dataset(s) used: MRPC (Microsoft Research Paraphrase Corpus)
- Domain: Natural language text (sentence pairs)
- Evidence: "MRPC: The Microsoft Research Paraphrase Corpus consists of sentence pairs and evaluates whether two sentences are semantically equivalent." (Section 4.1 Experimental Setup)

### Task: SST-2

- Task name: SST-2 (sentiment classification)
- Task type: Classification
- Dataset(s) used: SST-2 (Stanford Sentiment Treebank)
- Domain: Natural language text (movie review sentences)
- Evidence: "SST2: The Stanford Sentiment Treebank (binary classification version) uses single movie review sentences to assess sentiment." (Section 4.1 Experimental Setup)

### Task: QNLI

- Task name: QNLI (question-answering natural language inference)
- Task type: Classification
- Dataset(s) used: QNLI
- Domain: Natural language text (question + passage sentence)
- Evidence: "QNLI: The Question-answering Natural Language Inference dataset is a question paired with a sentence from a passage. The task is to determine if the sentence contains the answer to the question." (Section 4.1 Experimental Setup)

---

## 4. Domain and Modality Scope

- Is evaluation performed on a single domain? Yes, all tasks are text-based: "sentence pairs," "single movie review sentences," and "a question paired with a sentence from a passage." (Section 4.1 Experimental Setup)
- Multiple domains within the same modality? The paper evaluates multiple text datasets within GLUE: "MRPC ... SST-2 ... QNLI." (Section 4.1 Experimental Setup)
- Multiple modalities? Not stated.
- Does the paper claim domain generalization or cross-domain transfer? Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| MRPC | Not specified. | Not specified. | Not specified. | "We experiment with several datasets from GLUE, i.e. MRPC Dolan and Brockett [2005], SST-2 Socher et al. [2013], QNLI for training tasks." (Section 4.1 Experimental Setup) "All the model are trained from scratch for 30 epochs." (Section 4.1 Experimental Setup) |
| SST-2 | Not specified. | Not specified. | Not specified. | "We experiment with several datasets from GLUE, i.e. MRPC Dolan and Brockett [2005], SST-2 Socher et al. [2013], QNLI for training tasks." (Section 4.1 Experimental Setup) "All the model are trained from scratch for 30 epochs." (Section 4.1 Experimental Setup) |
| QNLI | Not specified. | Not specified. | Not specified. | "We experiment with several datasets from GLUE, i.e. MRPC Dolan and Brockett [2005], SST-2 Socher et al. [2013], QNLI for training tasks." (Section 4.1 Experimental Setup) "All the model are trained from scratch for 30 epochs." (Section 4.1 Experimental Setup) |

---

## 6. Input and Representation Constraints

- Fixed or variable input resolution? Not specified.
- Fixed patch size? Not specified.
- Fixed number of tokens? A maximum is stated: "max positions 512." (Section 4.1 Experimental Setup)
- Fixed dimensionality? "256-dimensional embeddings, 256-dimensional attention" is specified. (Section 4.1 Experimental Setup)
- Padding or resizing requirements? Not specified.
- Complex input representation form: "E_{\text{complex}}(x, \text{pos}) = E_{\text{vocab}}(x) + i \cdot E_{\text{pos}}(\text{pos})" (Section 3.1 Complex Encoding Layer)
- Positional component form: "we use sinusoidal encoding Vaswani et al. [2023] in imaginary part to encode position information" (Section 3.1 Complex Encoding Layer)

---

## 7. Context Window and Attention Structure

- Maximum sequence length: "max positions 512." (Section 4.1 Experimental Setup)
- Fixed or variable sequence length: Not specified beyond a maximum.
- Attention type: The paper states "phase-aware attention in the first layer of the transformer model" with "standard attention layers" afterward. (Abstract)
- Mechanisms to manage computational cost: "Our method applies phase-aware attention only to the first layer, followed by standard attention layers... limiting complex operations to one layer makes this encoding easy to adapt and maintains reasonable computational cost." (Section 3.4 Computation cost)
- Linear-attention compatibility: "We show that CoPE doesn't exhibit long term decay and is compatible with linear attention." (Abstract)

---

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: "Our approach replaces traditional positional encodings with complex embeddings where the real part captures semantic content and the imaginary part encodes positional information." (Abstract)
- Specific encoding used: "we use sinusoidal encoding Vaswani et al. [2023] in imaginary part to encode position information" (Section 3.1 Complex Encoding Layer)
- Where it is applied: "phase-aware attention in the first layer of the transformer model to capture position-dependent patterns, followed by standard attention layers for higher-levels." (Abstract)
- Comparisons/ablations: "Table 1: Test Performance comparison of different positional encodings across multiple datasets." (Section 4.1 Experimental Setup) and "we plot the training loss vs. number of epochs for different variant of CoPE i.e., CoPE magnitude, CoPE phase, CoPE hybrid-norm and compare with RoPE." (Section 4.2 Results)
- Fixed across experiments vs modified per task: Not stated explicitly; multiple positional encodings and CoPE variants are compared. (Section 4.1 Experimental Setup; Section 4.2 Results)

---

## 9. Positional Encoding as a Variable

- Core research variable: "We introduce CoPE (a lightweight Complex Positional Encoding), a novel architecture" (Abstract)
- Multiple positional encodings compared: "Table 1: Test Performance comparison of different positional encodings across multiple datasets." (Section 4.1 Experimental Setup)
- Variant comparisons: "different variant of CoPE i.e., CoPE magnitude, CoPE phase, CoPE hybrid-norm and compare with RoPE." (Section 4.2 Results)
- Claims that positional encoding is not critical/secondary: Not stated.

---

## 10. Evidence of Constraint Masking

- Model size(s): "We use transformer model with 6 layers, 8 heads, 256-dimensional embeddings, 256-dimensional attention with max positions 512." (Section 4.1 Experimental Setup)
- Dataset size(s): Not specified.
- Performance gains attributed to architecture/encoding rather than scaling: "Experimental evaluation on the GLUE benchmark suggest that our approach achieves superior performance with less computational complexity, compared to RoPE, Sinusoidal and Learned positional encodings." (Abstract)
- Computational scaling claim: "CoPE is L times faster than RoPE." (Section 3.4 Computation cost)
- Resource constraints noted: "Due to resource constraints, our current method is only evaluated on relatively smaller model that is trained from scratch." (Section 5 Limitations)

---

## 11. Architectural Workarounds

- Phase-aware attention only in the first layer: "We introduce phase-aware attention in the first layer of the transformer model to capture position-dependent patterns, followed by standard attention layers for higher-levels." (Abstract)
- Cost control by limiting complex ops: "Our method applies phase-aware attention only to the first layer... limiting complex operations to one layer makes this encoding easy to adapt and maintains reasonable computational cost." (Section 3.4 Computation cost)
- Keep values real to propagate to later layers: "We keep the value vector V in real space (projection on z_{real}), to propogate the real valued output to next layers." (Section 3.2 Phase-Aware Attention)
- Linear attention compatibility: "We show that CoPE doesn't exhibit long term decay and is compatible with linear attention." (Abstract)

---

## 12. Explicit Limitations and Non-Claims

- Extrapolation not fully evaluated: "We plan to include the extrapolation experiments with CoPE and compare with AliBi Press et al. [2022]." (Section 5 Limitations)
- Pretraining/finetuning not evaluated: "Due to resource constraints, our current method is only evaluated on relatively smaller model that is trained from scratch. In particular, CoPE requires a separate evaluation on pretraining and fine tuning tasks on larger models." (Section 5 Limitations)
- Explicit statements about not attempting open-world learning or unrestrained multi-task learning: Not stated.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: Text-only GLUE datasets with "sentence pairs," "single movie review sentences," and "a question paired with a sentence from a passage." (Section 4.1 Experimental Setup)
> – Task structure: Multiple supervised classification tasks (MRPC/SST-2/QNLI) listed as "datasets from GLUE." (Section 4.1 Experimental Setup)
> – Representation rigidity: Fixed embedding size and max positions ("256-dimensional embeddings"; "max positions 512") plus complex input form "E_{\text{complex}}(x, \text{pos}) = E_{\text{vocab}}(x) + i \cdot E_{\text{pos}}(\text{pos})." (Section 4.1 Experimental Setup; Section 3.1 Complex Encoding Layer)
> – Model sharing vs specialization: Not specified whether weights are shared; training details only say "All the model are trained from scratch for 30 epochs." (Section 4.1 Experimental Setup)
> – Role of positional encoding: Central variable with multiple encodings compared ("Test Performance comparison of different positional encodings across multiple datasets"). (Section 4.1 Experimental Setup)

---

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple tasks/datasets: "MRPC... SST-2... QNLI" (Section 4.1 Experimental Setup), which indicates multiple tasks rather than a single-task setting. All tasks are text-based (e.g., "sentence pairs," "single movie review sentences," and "a question paired with a sentence from a passage"), so the evaluation remains within a single modality/domain of natural language text. (Section 4.1 Experimental Setup)
