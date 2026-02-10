# Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (Not specified in the paper)
Source: Transformer-XL- Attentive Language Models Beyond a Fixed-Length Context.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Autoregressive language modeling (next-token prediction) | Corpus token sequences (word-level and character-level); context tokens `x_{<t}` | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Categorical probability distribution over the next token | 0D (inferred) | Fixed (inferred) |
| Long-form text generation (seeded continuation) | Seed context token sequence from WikiText-103 test set ("at most 512 consecutive tokens") | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Generated text token sequence / article continuation | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper covers token-sequence tasks in language: autoregressive language modeling across word-level and character-level corpora, plus long-form generation from a seed context. The supported modality is textual tokens in 1D temporal order, with outputs expressed either as per-step next-token distributions or extended generated token streams. From the architecture description, context handling is capped by configured segment/memory lengths, attention is static over a predefined span, and state is constructed via cached hidden-state reuse.

## Evidence
### Task: Autoregressive language modeling (next-token prediction)
- "Given a corpus of tokens  $\mathbf{x} = (x_1, \dots, x_T)$ , the task of language modeling is to estimate the joint probability  $P(\mathbf{x})$ , which is often auto-regressively factorized as  $P(\mathbf{x}) = \prod_t P(x_t \mid \mathbf{x}_{< t})$ ." (Section 3 Model)
- "The logits are then fed into the Softmax function, yielding a categorical probability distribution over the next token." (Section 3 Model)
- "During training, the hidden state sequence computed for the previous segment is fixed and cached to be reused as an extended context when the model processes the next new segment" (Section 3.2 Segment-Level Recurrence with State Reuse)
- Inference: `1D (t)` is inferred from token-sequence corpus/context; `Capped` is inferred from segmented processing plus predefined memory length ("we can cache a predefined length-M old hidden states"); `Static` attention is inferred because the model attends within predefined segment/memory spans; `Constructed` state is inferred from explicit cached hidden-state memory reuse; `0D`/`Fixed` output is inferred from per-step next-token distribution.

### Task: Long-form text generation (seeded continuation)
- "Transformer-XL is also able to generate relatively coherent long text articles with *thousands of* tokens (see Appendix E), trained on only 100M tokens." (Section 1 Introduction)
- "Trained only on WikiText-103 which is mediumsized, Transformer-XL is already able to generate relatively coherent articles with thousands of tokens without manual cherry picking, despite minor flaws." (Section 4.4 Generated Text)
- "we seed the our Transformer-XL with a context of at most 512 consecutive tokens randomly sampled from the test set of Wikitext-103. Then, we run Transformer-XL to generate a *pre-defined* number of tokens (500 or 1,000 in our case)." (Appendix E Generated Text)
- Inference: `1D (t)` is inferred for both seed and generated text streams; `Capped` input dynamics is inferred from the stated seed limit ("at most 512 consecutive tokens"); `Static` attention and `Constructed` state are inferred from the same Transformer-XL recurrence/memory mechanism described in Section 3.2; `Open` output dynamics is inferred because generation is autoregressive and described as extending to "thousands of" tokens with adjustable generation length.
