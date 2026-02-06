# End-to-End Test-Time Training for Long Context (Not specified in the paper)
Source: End-to-End Test-Time Training for Long Context.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next-token prediction (language modeling) | tokens (context sequence) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | token distribution (next token) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper focuses on long-context language modeling framed as next-token prediction over token sequences. Inputs are 1D token streams with capped context lengths in experiments (e.g., T=128K), and outputs are next-token distributions. The method uses a fixed sliding-window attention scheme and updates weights at test time, implying static attention and constructed state.

## Evidence
### Task: next-token prediction (language modeling)
- "Consider the standard task of next-token prediction, which consists of two phases at test time:" (Section 2 Method)
- "Prefill: conditioning on T+1 given tokens  $x_0, x_1, ..., x_T$" (Section 2 Method)
- "Decode: predicting a distribution  $\hat{p}_{T+1}$  over all possible instantiations of the next token." (Section 2 Method)
- "our model continues learning at test time via next-token prediction on the given context, compressing the context it reads into its weights." (Abstract)
- "our main method only restricts them to a fixed window size k." (Section 2.3)
- "Now information of  $x_1$  is stored in the updated MLPs (blue)." (Figure 2)
- Inference: Input is 1D (t) and output is 0D because the paper defines an ordered token sequence $x_0, x_1, ..., x_T$ and predicts a single next-token distribution; In Dynamics are Capped and Attention is Static because the method uses a fixed context length T and a fixed window size k; State is Constructed because context information is compressed into weights and stored in updated MLPs. (Section 2 Method; Section 2.3; Abstract; Figure 2)
