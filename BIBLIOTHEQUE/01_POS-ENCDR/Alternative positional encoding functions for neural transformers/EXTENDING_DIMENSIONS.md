## 1. Basic Metadata

- Title: "ALTERNATIVE POSITIONAL ENCODING FUNCTIONS FOR NEURAL TRANSFORMERS" (document header)
- Authors: "© Ezequiel López-Rubio*"; "Macorís Decena-Giménez"; "Rafael Marcos Luque-Baena" (author block)
- Year: 2025 ("December 23, 2025" in document header)
- Venue: Preprint ("A PREPRINT" in document header)

---

## 2. One-Sentence Contribution Summary

The paper proposes "an alternative set of periodic functions ... for positional encoding" that "preserve some key properties of sinusoidal ones, while they depart from them in fundamental ways" (Abstract).

---

## 3. Tasks Evaluated

- Task name: English–German machine translation of image descriptions (Multi30K)
  - Task type: Generation (machine translation)
  - Dataset(s) used: Multi30K English–German image–description dataset
  - Domain: natural language captions (text)
  - Evidence: "We trained and evaluated the model on the Multi30K English–German image–description dataset [Elliott et al., 2016], which provided parallel English–German captions aligned at the sentence level." (Section 3 Experiments) "All text was lowercased and tokenized at the word level using language–specific tokenizers." (Section 3 Experiments)

---

## 4. Domain and Modality Scope

- Single domain? Yes. Evidence: "We trained and evaluated the model on the Multi30K English–German image–description dataset... parallel English–German captions" (Section 3 Experiments).
- Multiple domains within the same modality? Not stated.
- Multiple modalities? Not stated.
- Domain generalization or cross-domain transfer claimed? Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| English–German caption translation (Multi30K) | Not applicable (single task; separate models per PE variant) | Not specified | Not specified | "For each positional encoding variant, we trained 10 models" (Section 3 Experiments). |

---

## 6. Input and Representation Constraints

- Sequence length and padding: "batches were dynamically padded to the maximum sequence length within each batch, with a maximum allowed length of 256 tokens." (Section 3 Experiments)
- Special tokens/padding: "Special tokens were added to each sequence, including <sos> and <eos> to mark sentence boundaries, <unk> for out–of–vocabulary words, and <pad> for sequence padding." (Section 3 Experiments)
- Tokenized text inputs: "Sentences were converted to sequences of token indices" (Section 3 Experiments).
- Positional index range: "each position m \in \{0, ..., L-1\}" (Section 2 Methodology).
- Fixed dimensionality: "d_{\rm model} = 512" (Section 3 Experiments).
- Fixed input resolution: Not specified.
- Fixed patch size: Not specified.
- Fixed number of tokens: Not specified beyond the 256-token maximum.

---

## 7. Context Window and Attention Structure

- Maximum sequence length: "maximum allowed length of 256 tokens" (Section 3 Experiments).
- Fixed or variable length: "dynamically padded to the maximum sequence length within each batch" (Section 3 Experiments).
- Attention type: Standard Transformer self-attention is described, but no windowing/sparsity is specified: "Transformer architectures rely on positional encodings to inject order information into sequences processed by permutation-invariant self-attention layers" (Section 2 Methodology).
- Mechanisms to manage computational cost (windowing, pooling, pruning): Not stated.

---

## 8. Positional Encoding (Critical Section)

- Mechanism used: Absolute, input-additive periodic encodings. Evidence: "The original Transformer uses a deterministic periodic encoding" and "The encoding is then added to the token embeddings x_m ... before being fed to the first self-attention layer." (Section 2 Methodology)
- Alternative functions compared: "added four interchangeable positional encoding functions: 1. Sinusoidal ... 2. Triangular ... 3. Square ... 4. Sawtooth" (Section 3 Experiments).
- Where applied: Input only (added to embeddings before the first self-attention layer). Evidence: "The encoding is then added to the token embeddings x_m ... before being fed to the first self-attention layer." (Section 2 Methodology)
- Fixed vs modified: Positional encoding is varied across experiments (multiple functions compared). Evidence: "To systematically compare the four positional encoding variants, we employed 10–fold cross–validation" (Section 3 Experiments).

---

## 9. Positional Encoding as a Variable

- Core research variable? Yes. Evidence: "In this work, an alternative set of periodic functions is proposed for positional encoding." (Abstract)
- Multiple positional encodings compared? Yes. Evidence: "added four interchangeable positional encoding functions" (Section 3 Experiments).
- PE choice claimed as non-critical or secondary? Not stated.

---

## 10. Evidence of Constraint Masking

- Model size(s): "d_{\rm model} = 512, N = 6 layers in both encoder and decoder, h = 8 attention heads, ... d_{\rm ff} = 2048" (Section 3 Experiments).
- Dataset size(s): "the Multi30K training split, which consists of 29,001 sentence pairs" (Section 3 Experiments).
- Performance gains attributed to scaling model/data vs architecture/PE: The paper attributes gains to alternative positional encodings: "all three alternative functions clearly outperform the standard sinusoidal function" (Section 4 Discussion).
- Claims about scaling model size or data? Not stated.

---

## 11. Architectural Workarounds

- Not stated beyond using the standard Transformer base configuration: "The model architecture followed the Transformer base configuration" (Section 3 Experiments).

---

## 12. Explicit Limitations and Non-Claims

- Limitation / preliminary scope: "Some tentative experiments are reported" (Abstract).
- Future work / limitations: "Future work includes more experimentation with a wider range of problems so that the preliminary results presented here are further validated. Other alternative periodic functions may also be considered." (Section 4 Discussion)
- Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: Single dataset of English–German captions (text-only evaluation).
> – Task structure: Single sequence-to-sequence translation task with BLEU-4 evaluation.
> – Representation rigidity: Tokenized word sequences with dynamic padding and a 256-token maximum; fixed d_model.
> – Model sharing vs specialization: Separate model runs per positional encoding variant; no multi-task sharing.
> – Role of positional encoding: Central experimental variable; multiple fixed periodic functions compared.

---

### 14. Final Classification

**Single-task, single-domain.** The evaluation is on one task—English–German caption translation—using a single dataset: "We trained and evaluated the model on the Multi30K English–German image–description dataset... parallel English–German captions" (Section 3 Experiments). No multi-domain or multi-modality evaluation is described, and the experimental variation is in positional encoding functions rather than tasks or domains.
