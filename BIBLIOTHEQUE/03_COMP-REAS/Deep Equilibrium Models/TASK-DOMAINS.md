# Deep Equilibrium Models (Not specified in the paper)
Source: Deep Equilibrium Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequence prediction (copy memory) | 1D symbol sequence (integers 1-9 with zeros; delimiter 9) | 1D (t) | Fixed | Static (inferred) | Constructed (inferred) | symbol sequence (zeros then copied first 10 symbols) | 1D (t) | Fixed |
| language modeling (word-level) | word token sequence (word embeddings) | 1D (t) | Fixed | Static (inferred) | Constructed (inferred) | word token sequence predictions (inferred) | 1D (t) | Fixed |

## Summary
The paper evaluates DEQ on two sequence prediction tasks: a synthetic copy-memory sequence copying task and word-level language modeling (PTB and WikiText-103). Both tasks operate on 1D temporal sequences and are evaluated with fixed-length sequences in the experiments. Attention is treated as static and the model constructs internal equilibrium states, both inferred from the full-sequence formulation and equilibrium hidden state description.

## Evidence
### Task: sequence prediction (copy memory)
- "each sequence x_{1:(T+20)} is 1-dimensional and has length T+20" (Section F Task Descriptions)
- "The goal of this task is to produce y_{1:(T+20)}" (Section F Task Descriptions)
- "y_{T+11:T+20} = x_{1:10}" (Section F Task Descriptions)
- Inference: Attention Dynamic = Static and State Dynamic = Constructed because the model maps the full input sequence to an output sequence and defines an equilibrium hidden sequence; see "Given an input sequence x_{1:T} = [x_1, ..., x_T]" (Section 2) and "Let z_{1:T}* be an equilibrium hidden sequence" (Appendix A).

### Task: language modeling (word-level)
- "we evaluate the DEQ-TrellisNet instantiation on word-level language modeling with the PTB corpus." (Section 5.2)
- "On large-scale language modeling tasks, such as the WikiText-103 benchmark" (Abstract)
- "x_i in R^p (e.g., a word embedding)" (Section 2)
- "produces output G(x_{1:T}) = y_{1:T}" (Section 2)
- "with sequences of length 150 at both training and inference on the WikiText-103 dataset." (Section 5 Setting)
- Inference: Output is word-token predictions, and Attention Dynamic = Static and State Dynamic = Constructed, because the task is word-level language modeling over token sequences and the model forms an equilibrium hidden sequence; see "word-level language modeling" (Section 5.2) and "Let z_{1:T}* be an equilibrium hidden sequence" (Appendix A).
