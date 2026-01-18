## 1. Basic Metadata

- Title: "Self-Attention with Relative Position Representations" (Title block)
- Authors: "Peter Shaw Google petershaw@google.com" (Title block); "Jakob Uszkoreit Google Brain usz@google.com Ashish Vaswani Google Brain avaswani@google.com" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

"In this work we present an alternative approach, extending the self-attention mechanism to efficiently consider representations of the relative positions, or distances between sequence elements." (Abstract)

## 3. Tasks Evaluated

Task name: WMT 2014 English-to-German translation
- Task type: Generation
- Dataset(s): "WMT 2014 English-German dataset" (Section 4.1 Experimental Setup)
- Domain: natural language machine translation
- Quotes: "On the WMT 2014 English-to-German and English-to-French translation tasks" (Abstract); "We evaluated our model on the WMT 2014 machine translation task, using the WMT 2014 English-German dataset" (Section 4.1 Experimental Setup)

Task name: WMT 2014 English-to-French translation
- Task type: Generation
- Dataset(s): "2014 WMT English-French dataset" (Section 4.1 Experimental Setup)
- Domain: natural language machine translation
- Quotes: "On the WMT 2014 English-to-German and English-to-French translation tasks" (Abstract); "the 2014 WMT English-French dataset" (Section 4.1 Experimental Setup)

## 4. Domain and Modality Scope

- Evaluation scope: Single modality (machine translation sequences) across two language pairs: "On the WMT 2014 English-to-German and English-to-French translation tasks" (Abstract).
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| WMT 2014 English-to-German translation | Not specified. | Not specified. | Not specified. | "We evaluated our model on the WMT 2014 machine translation task, using the WMT 2014 English-German dataset" (Section 4.1 Experimental Setup) |
| WMT 2014 English-to-French translation | Not specified. | Not specified. | Not specified. | "the 2014 WMT English-French dataset" (Section 4.1 Experimental Setup) |

## 6. Input and Representation Constraints

- Fixed or variable input resolution? Not specified; input is a sequence with length n: "Each attention head operates on an input sequence, x = (x_1, \ldots, x_n) of n elements" (Section 2.2 Self-Attention).
- Fixed patch size? Not specified.
- Fixed number of tokens? Not specified per sequence; batch and vocabulary constraints are stated: "For all experiments, we split tokens into a 32,768 word-piece vocabulary" and "limited input and output tokens per batch to 4096 per GPU" (Section 4.1 Experimental Setup).
- Fixed dimensionality (e.g., strictly 2D)? Linear sequence assumption: "For linear sequences, edges can capture information about the relative position differences between input elements." (Section 3.2 Relative Position Representations).
- Any padding or resizing requirements? Not specified.
- Other explicit constraints on representation: "The maximum relative position we consider is clipped to a maximum absolute value of k." (Section 3.2 Relative Position Representations); "Therefore, we consider 2k+1 unique edge labels." (Section 3.2 Relative Position Representations).

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Sequence length fixed or variable: Variable n is assumed in the formulation: "input sequence, x = (x_1, \ldots, x_n) of n elements" (Section 2.2 Self-Attention).
- Attention type: Global self-attention over all positions: "z_i = \sum_{j=1}^n \alpha_{ij}(x_j W^V)" (Section 2.2 Self-Attention).
- Computational cost management: "we reduce the space complexity of storing relative position representations from O(hn^2d_a) to O(n^2d_a) by sharing them across each heads" (Section 3.3 Efficient Implementation); "both issues can be resolved by splitting the computation of eq. (4) into two terms" (Section 3.3 Efficient Implementation).

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: Relative position representations over clipped distances: "extending the self-attention mechanism to efficiently consider representations of the relative positions, or distances between sequence elements" (Abstract); "The maximum relative position we consider is clipped to a maximum absolute value of k" (Section 3.2 Relative Position Representations).
- Where applied: In self-attention compatibility and values: "We modify eq. (1) to propagate edge information to the sublayer output" and "We also, importantly, modify eq. (2) to consider edges when determining compatibility" (Section 3.1 Relation-aware Self-Attention).
- Baseline absolute positional encoding: "Position encodings based on sinusoids of varying frequency are added to encoder and decoder input elements prior to the first layer." (Section 2.1 Transformer).
- Layer usage: "When using relative position encodings, we used clipping distance k = 16, and used unique edge representations per layer and head." (Section 4.1 Experimental Setup); "used unique edge representations per layer." (Section 4.1 Experimental Setup).
- Fixed vs modified per task/experiment: "We compared our model using only relative position representations to the baseline Transformer (Vaswani et al., 2017) with sinusoidal position encodings." (Section 4.2 Machine Translation); "We evaluated the effect of varying the clipping distance, k" (Section 4.3 Model Variations); "We also evaluated the impact of ablating each of the two relative position representations" (Section 4.3 Model Variations).

## 9. Positional Encoding as a Variable

- Core research variable vs fixed assumption: Core variable, emphasized in the contribution and experiments: "we present an alternative approach, extending the self-attention mechanism to efficiently consider representations of the relative positions" (Abstract); "We compared our model using only relative position representations to the baseline Transformer (Vaswani et al., 2017) with sinusoidal position encodings." (Section 4.2 Machine Translation).
- Multiple positional encodings compared: Yes; relative vs absolute and combined: "we did not observe any benefit from including sinusoidal position encodings in addition to relative position representations." (Section 4.2 Machine Translation).
- PE choice claimed non-critical or secondary? No explicit claim; only the negative result above is stated.

## 10. Evidence of Constraint Masking

- Model sizes: Base and big configurations are specified: "For our base model, we used 6 encoder and decoder layers, d_x = 512, d_z = 64, 8 attention heads" and "For our big model, we used 6 encoder and decoder layers, d_x=1024, d_z=64, 16 attention heads" (Section 4.1 Experimental Setup).
- Dataset sizes: "WMT 2014 English-German dataset consisting of approximately 4.5M sentence pairs" and "2014 WMT English-French dataset consisting of approximately 36M sentence pairs" (Section 4.1 Experimental Setup).
- Attribution of gains: Improvements are attributed to relative position representations, not scaling or data size: "this approach yields improvements of 1.3 BLEU and 0.3 BLEU over absolute position representations" (Abstract); "our approach improved performance over our baseline" (Section 4.2 Machine Translation).
- Scaling model size/data as primary driver: Not explicitly stated.

## 11. Architectural Workarounds

- Clipped relative positions to limit edge labels: "The maximum relative position we consider is clipped to a maximum absolute value of k." (Section 3.2 Relative Position Representations).
- Shared relative position representations across heads to reduce memory: "we reduce the space complexity of storing relative position representations from O(hn^2d_a) to O(n^2d_a) by sharing them across each heads." (Section 3.3 Efficient Implementation).
- Efficient computation via decomposition of attention term: "both issues can be resolved by splitting the computation of eq. (4) into two terms" (Section 3.3 Efficient Implementation).

## 12. Explicit Limitations and Non-Claims

- "However, as explored in 4.3, this may not be necessary for machine translation." (Section 3.1 Relation-aware Self-Attention).
- "Including relative position representations solely when determining compatibility between elements may be sufficient, but further work is needed to determine whether this is true for other tasks." (Section 4.3 Model Variations).
- "For future work, we plan to extend this mechanism to consider arbitrary directed, labeled graph inputs to the Transformer. We are also interested in nonlinear compatibility functions to combine input representations and edge representations." (Section 5 Conclusions).
- "Notably, we observe that combining relative and absolute position representations yields no further improvement in translation quality." (Abstract).

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: "WMT 2014 English-to-German and English-to-French translation tasks" (Abstract) indicate a machine translation domain.
> - Task structure: Two translation datasets are evaluated: "WMT 2014 English-German dataset" and "2014 WMT English-French dataset" (Section 4.1 Experimental Setup).
> - Representation rigidity: Linear sequences with clipped relative positions: "For linear sequences" and "clipped to a maximum absolute value of k" (Section 3.2 Relative Position Representations); fixed subword vocabulary: "32,768 word-piece vocabulary" (Section 4.1 Experimental Setup).
> - Model sharing vs specialization: Not specified.
> - Role of positional encoding: Central experimental variable with comparisons and ablations: "We compared our model using only relative position representations to the baseline Transformer ... with sinusoidal position encodings" (Section 4.2 Machine Translation); "We also evaluated the impact of ablating each of the two relative position representations" (Section 4.3 Model Variations).

### 14. Final Classification

**Multi-task, single-domain**.
The evaluation covers two machine translation tasks: "WMT 2014 English-to-German and English-to-French translation tasks" (Abstract), with separate datasets listed in Section 4.1. The paper does not report multi-domain or cross-modal evaluation, so the evidence supports multi-task within a single domain.
