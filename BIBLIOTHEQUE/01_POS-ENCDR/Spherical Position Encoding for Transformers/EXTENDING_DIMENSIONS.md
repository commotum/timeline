## 1. Basic Metadata

Title: SPHERICAL POSITION ENCODING FOR TRANSFORMERS
Authors: Eren Unlu
Year: Year not specified.
Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

"In this paper, we introduce the notion of \"geotokens\"" and "formulate a position encoding mechanism based on RoPE architecture which is adjusted for spherical coordinates" to encode geographical coordinates rather than sequential positions. (Abstract)

---

## 3. Tasks Evaluated

Task name: Not specified (no evaluation tasks are described in the OCR text).
Task type: Not specified.
Dataset(s) used: Not specified.
Domain: Not specified.
Evidence: "In this work, we have presented a novel concept for the transformer architecture, integrating \"geotokens\" as a representation of geographical entities." (Conclusion)

---

## 4. Domain and Modality Scope

- Evaluation domain scope: Not specified (no evaluation/datasets reported).
- Domain/modality described: Geographical entities with latitude/longitude; embeddings may come from different modalities.
- Domain generalization or cross-domain transfer: Not claimed.

Evidence:
- "A geotoken encapsulates both the semantic meaning and the spatial information of a geographical entity." (Section 2 Geotokens and Cartographical Transformer)
- "For the sake of simplicity only punctual locations are considered represented by a latitude and longitude." (Section 2 Geotokens and Cartographical Transformer)
- "It is assumed that each data point has a pre-embedded vector retaining valuable information about the location itself, which may have been encoded by any type of neural architecture or mechanism, such as a natural language model processing its verbal description or a CNN extracting its visual features." (Section 2 Geotokens and Cartographical Transformer)

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Not specified (no tasks evaluated) | Not specified. | Not specified. | Not specified. | "In this work, we have presented a novel concept for the transformer architecture, integrating \"geotokens\" as a representation of geographical entities." (Conclusion) |

---

## 6. Input and Representation Constraints

- Input representation: Geographical coordinates with latitude/longitude; geotokens have pre-embedded vectors.
  - "For the sake of simplicity only punctual locations are considered represented by a latitude and longitude." (Section 2 Geotokens and Cartographical Transformer)
  - "It is assumed that each data point has a pre-embedded vector retaining valuable information about the location itself, which may have been encoded by any type of neural architecture or mechanism, such as a natural language model processing its verbal description or a CNN extracting its visual features." (Section 2 Geotokens and Cartographical Transformer)
- Fixed dimensionality: Embedding dimension constrained to multiples of three; padding suggested as a possible workaround.
  - "For the sake of simplicity, the embedding dimension is a multiple of three due to natural requirements, however this choice might be unconvenient as many embedders of different modalities might not adhere to this constraint. The possible circumvention to this issue is out of scope of this paper, such as possibly adding padding indices." (Section 5 Spherical Position Encoding)
- Spherical Earth assumption: "let us assume that globe is a perfect sphere with constant radius R." (Section 5 Spherical Position Encoding)
- Fixed/variable input resolution, fixed patch size, fixed number of tokens: Not specified.

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Fixed or variable length: Not specified.
- Attention type: Not specified (only general self-attention is mentioned).
- Computational cost mechanisms: Not specified.

Evidence:
- "parallel processing for efficiency with self-attention" (Section 1 Introduction)

---

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: RoPE-style rotary encoding extended to spherical coordinates.
  - "formulate a position encoding mechanism based on RoPE architecture which is adjusted for spherical coordinates." (Abstract)
  - "we propose to extend the RoPE method in spherical coordinates." (Section 5 Spherical Position Encoding)
- Where applied: In RoPE formulation to query/key projections via rotation matrix.
  - "\langle f_q(x_m, m), f_k(x_n, n) \rangle = g(x_m, x_n, m - n)" and "f_{\{a,k\}}(x_m,m) = \mathbf{R}_{\Theta,m}^d \mathbf{W} x_m" (Section 4 Rotary Position Embedding (RoPE))
- Fixed vs modified per task / ablated: Not specified (no task-specific variations or ablations reported).

---

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption: Core research variable (central contribution is a new PE).
  - "In this paper, we introduce the notion of \"geotokens\"... formulate a position encoding mechanism based on RoPE architecture which is adjusted for spherical coordinates." (Abstract)
- Multiple positional encodings compared: Not specified (no comparisons reported).
- PE claimed as not critical/secondary: Not claimed.

---

## 10. Evidence of Constraint Masking

- Model size(s): Not specified.
- Dataset size(s): Not specified.
- Performance gains attributed to scaling/architecture/tricks: Not specified (no results or ablations reported).

---

## 11. Architectural Workarounds

- Spherical rotation-based positional encoding to handle geographic coordinates.
  - "we propose to extend the RoPE method in spherical coordinates." (Section 5 Spherical Position Encoding)
- Dimensionality constraint with possible padding suggested as out-of-scope workaround.
  - "the embedding dimension is a multiple of three due to natural requirements... The possible circumvention to this issue is out of scope of this paper, such as possibly adding padding indices." (Section 5 Spherical Position Encoding)
- No windowed attention, hierarchical stages, or token pooling described.

---

## 12. Explicit Limitations and Non-Claims

- Simplifying assumptions about representation and geometry:
  - "For the sake of simplicity only punctual locations are considered represented by a latitude and longitude." (Section 2 Geotokens and Cartographical Transformer)
  - "let us assume that globe is a perfect sphere with constant radius R." (Section 5 Spherical Position Encoding)
- Out-of-scope limitations:
  - "The possible circumvention to this issue is out of scope of this paper, such as possibly adding padding indices." (Section 5 Spherical Position Encoding)
  - "In addition, further possible challenges such as proper scaling are kept out of scope as well, where in case one training the architecture with limited geolocations, rather than whole globe." (Section 5 Spherical Position Encoding)
- Explicit non-claims about open-world or multi-task learning: Not stated.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: Geographic entities represented as "geotokens" with latitude/longitude; no evaluation across domains reported.
> – Task structure: No explicit tasks, datasets, or evaluations are described; the paper is conceptual.
> – Representation rigidity: Assumes spherical Earth and embedding dimension as a multiple of three (padding out of scope).
> – Model sharing vs specialization: No multi-task setup or sharing details described.
> – Role of positional encoding: Central contribution is a spherical RoPE-style positional encoding.

---

### 14. Final Classification

**Single-task, single-domain**

Justification: The paper focuses on geographical entities and coordinate-based positional encoding, e.g., "geotokens" that represent "a latitude and longitude" for locations. (Section 2 Geotokens and Cartographical Transformer) No explicit multi-task or multi-domain evaluation is reported, and no datasets or tasks are described.
