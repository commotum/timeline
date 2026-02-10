# Repeated examples help learn arithmetic (Not specified in the paper.)
Source: Repeated Examples Help Learn Arithmetic.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCD prediction | integer token sequences representing two integers | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | integer token sequence representing GCD | 1D (t) (inferred) | Capped (inferred) |
| Modular multiplication prediction (mod 67) | integer token sequences representing two integers | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | integer token sequence representing product mod 67 | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates one model family on two arithmetic prediction tasks: greatest common divisor and modular multiplication. In both cases, the OCR explicitly describes integer inputs/outputs tokenized as digit sequences, which supports 1D (t) input/output domains with bounded (Capped) interfaces. The architecture described is a standard sequence-to-sequence transformer, so Attention is labeled Static (inferred) and State is labeled Direct (inferred) based on the absence of explicit runtime retrieval/control or persistent constructed state mechanisms in the task setup.

## Evidence
### Task: GCD prediction
- "In the greatest common divisor problem, the model is tasked to predict the GCD of two integers uniformly distributed between 1 and 1 million, encoded in base 1000." (Section 2 Experimental settings)
- "The integer inputs and outputs of both problems are tokenized as sequences of digits in base 1000, preceded by a sign which serves as a separator." (Section 2 Experimental settings)
- "We use sequence-to-sequence transformers (Vaswani et al., 2017) with 4 layers in the encoder and decoder, an embedding dimension of 512, and 8 attention heads (35 million trainable parameters)." (Section 2 Experimental settings)
- Inference: In/Out Dimension are marked 1D (t) and In/Out Dynamics are marked Capped because the paper specifies sequence tokenization and bounded integer ranges ("between 1 and 1 million"). Attention Dynamic is marked Static and State Dynamic is marked Direct because the task description uses a standard seq2seq transformer interface without explicit runtime selection/retrieval control or externally maintained constructed state (Section 2 Experimental settings).

### Task: Modular multiplication prediction (mod 67)
- "In modular multiplication, we train models to predict the product, modulo 67, of two integers between 1 and a million." (Section 2 Experimental settings)
- "The integer inputs and outputs of both problems are tokenized as sequences of digits in base 1000, preceded by a sign which serves as a separator." (Section 2 Experimental settings)
- "Models are trained to minimize a cross-entropy loss, using the Adam optimizer (Kingma & Ba, 2014), with a learning rate of  $10^{-5}$ , and batches of 64." (Section 2 Experimental settings)
- Inference: In/Out Dimension are marked 1D (t) and In/Out Dynamics are marked Capped based on digit-sequence tokenization and bounded arithmetic domains (inputs capped at 1..1,000,000 and outputs constrained by modulo 67). Attention Dynamic is marked Static and State Dynamic is marked Direct because the OCR describes a standard seq2seq setup and does not specify dynamic runtime information selection or persistent constructed task state (Section 2 Experimental settings).
