# Scaling Laws and Interpretability of Learning from Repeated Data (Not specified in the paper)
Source: Scaling Laws and Interpretability of Learning from Repeated Data.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Next-token prediction and generation (natural language) | text tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | next text tokens | 1D (t) (inferred) | Capped (inferred) |
| Next-token prediction and generation (Python code) | Python code tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | next code tokens | 1D (t) (inferred) | Capped (inferred) |
| In-context sequence copying (repeated paragraph continuation) | repeated text tokens (Harry Potter paragraph copies) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | copied continuation tokens | 1D (t) (inferred) | Capped (inferred) |
| Prefix matching evaluation (induction-head behavior) | repeated random token sequences | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | prefix matching score | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers autoregressive token prediction/generation on natural language and Python code, plus two targeted evaluations of sequence behavior: copying and prefix matching. All identified tasks operate over token sequences, so the justified input structure is 1D (t), with capped dynamics inferred from the stated 8192-token context limit. The paper does not describe runtime retrieval or external control policies, so attention is classified as Static (inferred), and the task state is classified as Direct (inferred) for these reactive next-token mappings. Outputs are token sequences for modeling/copying and a scalar score for prefix matching.

## Evidence
### Task: Next-token prediction and generation (natural language)
- "The decoder-only transformer models were trained on an 8192 token context with the same settings as described in [Askell et al., 2021] for 100B tokens." (Section 3 Methods)
- "Our language experiments utilized a 400B token dataset with 55% heavily filtered common crawl data (220B tokens), 32% internet books (128B tokens), and some smaller distributions including OpenWebText, Wikipedia, and Stack Exchange; most of which we sourced from The Pile [Gao et al., 2021], and leveraged the 50,304 vocabulary GPT-2 encoding [Radford et al., 2019, Wolf et al., 2019]." (Section 3 Methods)
- Inference: `1D (t)` and `Capped` are inferred from token-sequence modeling plus the explicit "8192 token context"; `Static` attention and `Direct` state are inferred from standard decoder-only next-token setup described in Section 3 with no runtime input-selection mechanism specified.

### Task: Next-token prediction and generation (Python code)
- "Code models were trained or fine-tuned on 45B tokens of Python for 2.2 epochs." (Section 3 Methods)
- "Training on repeated Python code creates a similar behavior. When training on Python we also observe a double descent phenomenon and a predictable poor performance region in terms of model size and repeated epochs, though the shape of both curves are somewhat different." (Section 1.1 Summary of Results)
- Inference: `1D (t)` input/output and `Capped` dynamics are inferred from tokenized Python modeling under the same transformer context setting; `Static` attention and `Direct` state are inferred for the same reason as the natural-language next-token task.

### Task: In-context sequence copying (repeated paragraph continuation)
- "We constructed a simple copying eval, the loss on the first paragraph of Harry Potter copied 11 times." (Section 1.1 Summary of Results)
- "The ability of a language model to copy text (in the sense of being provided with a context consisting of a passage repeated several times, and testing whether the model can repeat it once more) is a potential measure of *generalization*, as copying is independent of the content of the text." (Section 2 Results)
- Inference: The copying setup is modeled as `1D (t)` token-sequence input/output with `Capped` dynamics due the model's finite context window; `Static` attention and `Direct` state are inferred because the evaluation uses standard forward passes over provided context.

### Task: Prefix matching evaluation (induction-head behavior)
- "In line with [Olsson et al., 2022] we evaluated the models on their prefix matching score, repeated sequences of random tokens and observed the degree to which attention heads attend to earlier tokens that are preceded by a token that matches the present token." (Section 1.1 Summary of Results)
- "We decided to probe the prefix matching score as measure of mechanistic structure that is distinct from the behavior of copying itself." (Section 2 Results)
- Inference: Input is treated as `1D (t)` with `Capped` dynamics (token sequences under fixed context); output is labeled `0D` and `Fixed` because the reported object is a scalar "prefix matching score" per evaluation setting; `Static` attention and `Direct` state are inferred from the same standard transformer evaluation setup.
