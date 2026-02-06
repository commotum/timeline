# CLOSED-LOOP TRANSFORMERS: AUTOREGRESSIVE MODELING AS ITERATIVE LATENT EQUILIBRIUM (2025)
Source: Autoregressive Modeling as Iterative Latent Equilibrium (Equilibrium Transformers, EqT).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Next-token prediction (autoregressive language modeling) | tokens (context) | 1D (t) | Capped (inferred) | Static (inferred) | Constructed | next token | 0D | Fixed |
| Binary cumulative parity prediction | bit sequence | 1D (t) | Capped | Static (inferred) | Constructed | cumulative XOR bits | 1D (t) | Capped |

## Summary
The paper centers on autoregressive next-token prediction over token sequences and evaluates the architecture on the binary cumulative parity sequence prediction task. Both tasks operate on 1D sequences, with input length treated as bounded by a context window or specified length range, and the parity outputs are likewise capped by sequence length. The model constructs latent state via iterative refinement; attention is treated as static over the provided context (inferred from the fixed context formulation).

## Evidence
### Task: Next-token prediction (autoregressive language modeling)
- "at each time step t, the model computes a single forward pass to produce a hidden state  $\mathbf{h}_t$  and emits a token distribution" (Section 2.1)
- "$p(\mathbf{x}_{t+1} \mid \mathbf{x}_{\leq t})$ ." (Section 2.1)
- "The final layer's output  $\mathbf{h}_t^{(L)}$  is passed to a language modeling head  $p_{\mathrm{LM}}(\cdot \mid \mathbf{h}_t^{(L)})$  to produce next-token probabilities." (Section 3.4)
- "before emitting any token, the model must iteratively refine its latent representation until it reaches a self-consistent equilibrium" (Introduction)
- Inference: Input dynamics and attention are marked as capped/static because the model operates over a "context window" and applies "\\text{MultiHeadAttention}(\\mathbf{h}_t, \\mathbf{x}_{\\leq t})" to the given context. (Section 3.3.4; Algorithm 1)

### Task: Binary cumulative parity prediction
- "#### 5.1.1 Task: Binary Cumulative Parity" (Section 5.1.1)
- "Given an input sequence of bits  $\mathbf{x} = [b_1, b_2, \dots, b_n]$  where  $b_i \in \{0, 1\}$" (Section 5.1.1)
- "the model must predict the cumulative XOR at each position:" (Section 5.1.1)
- "We test sequence lengths  $L\in\{8,16,32,48,64,96,128,192,256\}$ ." (Section 5.1.2)
- Inference: Attention is marked static because the parity experiments use standard transformer self-attention over the fixed input sequence ("8 attention heads"). State is marked constructed because EqT "iteratively refine[s] its latent representation" before output. (Section 5.1.2; Introduction)
