# Self-Attention with Relative Position Representations (Year not specified in the paper.)
Source: Self-Attention with Relative Position Representations.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Machine translation (English-to-German) | English source token sequence | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | German target token sequence | 1D (t) | Capped (inferred) |
| Machine translation (English-to-French) | English source token sequence | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | French target token sequence | 1D (t) | Capped (inferred) |

## Summary
The paper evaluates text-to-text machine translation on two tasks: English-to-German and English-to-French. Both tasks operate on token sequences and produce token sequences, supporting 1D (t) input and output dimensions. Attention is inferred as Static from the formulation that aggregates over all sequence positions, while state is inferred as Direct because outputs are computed as weighted combinations of transformed input elements. Input and output dynamics are inferred as Capped based on the explicit token processing limits used in experiments.

## Evidence
### Task: Machine translation (English-to-German)
- "On the WMT 2014 English-to-German and English-to-French translation tasks, this approach yields improvements of 1.3 BLEU and 0.3 BLEU over absolute position representations, respectively." (Abstract)
- "using the WMT 2014 English-German dataset consisting of approximately 4.5M sentence pairs" (Section 4.1 Experimental Setup)
- Inference: Input/Output dimension is 1D (t) because "Each attention head operates on an input sequence,  $x = (x_1, \ldots, x_n)$  of n elements where  $x_i \in \mathbb{R}^{d_x}$ , and computes a new sequence  $z = (z_1, \ldots, z_n)$  of the same length where  $z_i \in \mathbb{R}^{d_z}$ ." (Section 2.2 Self-Attention). Attention Dynamic is Static because "$$z_i = \sum_{j=1}^n \alpha_{ij}(x_j W^V) \tag{1}$$" uses a predefined full sequence range (Section 2.2 Self-Attention). In/Out Dynamics are Capped from "limited input and output tokens per batch to 4096 per GPU" (Section 4.1 Experimental Setup). State Dynamic is Direct because "Each output element,  $z_i$ , is computed as weighted sum of a linearly transformed input elements:" (Section 2.2 Self-Attention).

### Task: Machine translation (English-to-French)
- "On the WMT 2014 English-to-German and English-to-French translation tasks, this approach yields improvements of 1.3 BLEU and 0.3 BLEU over absolute position representations, respectively." (Abstract)
- "and the 2014 WMT English-French dataset consisting of approximately 36M sentence pairs." (Section 4.1 Experimental Setup)
- Inference: Input/Output dimension is 1D (t) because "Each attention head operates on an input sequence,  $x = (x_1, \ldots, x_n)$  of n elements where  $x_i \in \mathbb{R}^{d_x}$ , and computes a new sequence  $z = (z_1, \ldots, z_n)$  of the same length where  $z_i \in \mathbb{R}^{d_z}$ ." (Section 2.2 Self-Attention). Attention Dynamic is Static because "$$z_i = \sum_{j=1}^n \alpha_{ij}(x_j W^V) \tag{1}$$" uses a predefined full sequence range (Section 2.2 Self-Attention). In/Out Dynamics are Capped from "limited input and output tokens per batch to 4096 per GPU" (Section 4.1 Experimental Setup). State Dynamic is Direct because "Each output element,  $z_i$ , is computed as weighted sum of a linearly transformed input elements:" (Section 2.2 Self-Attention).
