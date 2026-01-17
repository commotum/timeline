## 1. Basic Metadata

- Title: "Exploring Context Window of Large Language Models via Decomposed Positional Vectors" (Title)
- Authors: "Zican Dong<sup>1</sup>\*, Junyi Li<sup>3</sup>\*, Xin Men<sup>4</sup>, Wayne Xin Zhao<sup>1</sup>†, Bingning Wang<sup>4</sup> Zhen Tian<sup>1</sup>, Weipeng Chen<sup>4</sup>, Ji-Rong Wen<sup>1,2</sup>" (Title block)
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

The paper claims to "explore the positional information within and beyond the context window" and to design "two training-free context window extension methods, positional vector replacement and attention window extension" to extend LLM context windows (Abstract).

## 3. Tasks Evaluated

- Task name: Language modeling (perplexity across length); Task type: Generation; Dataset(s): PG-19; Domain: Text (language modeling); Evidence: "we evaluate language modeling performance on the test set of PG-19 [22]" and "we measure PPL across various input lengths (from 2K to 8K) using a sliding window approach" (Section 4.3 Results on Language Modeling).
- Task name: Length extrapolation / positional-vector analysis with PPL; Task type: Generation; Dataset(s): RedPajama; Domain: Text (language modeling); Evidence: "We subsample 32K samples with the same number of tokens from RedPajama. We perform the inference on these data to obtain hidden states of LLMs" (Section 3.1 Experimental Settings) and "perform inference on samples consisting of 8192 tokens. Further, we analyze the change in PPL score" (Section 3.3.1 Direct Extrapolation).

## 4. Domain and Modality Scope

- Evaluation is performed on a single domain and single modality (text), supported by "processing text beyond the length of the context window" (Abstract), "language modeling performance on the test set of PG-19 [22]" (Section 4.3 Results on Language Modeling), and "We subsample 32K samples with the same number of tokens from RedPajama" (Section 3.1 Experimental Settings).
- Multiple domains within the same modality: Not indicated; datasets are text-only as above.
- Multiple modalities: Not indicated.
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Language modeling on PG-19 | Yes (same pretrained model variants evaluated) | No | Not specified | "two training-free context window extension methods" (Abstract) and "generalize to longer texts without fine-tuning" (Introduction) |
| RedPajama inference / PPL analyses | Yes (same pretrained model variants evaluated) | No | Not specified | "We continually pre-train the TinyLlama-1.1B checkpoint [23] on 50B tokens from RedPajama [24] with a context window C=2048, resulting in a set of comparison models with different positional encodings and attention mechanisms" (Section 3.1 Experimental Settings) and "training-free context window extension methods" (Abstract) |

## 6. Input and Representation Constraints

- Context window size is fixed during training: "with a context window C=2048" (Section 3.1 Experimental Settings) and "The context window size C is 2048" (Table 4).
- Input is token sequences with variable length T: "given an input sequence s of T tokens, i.e., {x_1, \ldots, x_T}" (Section 2 Background).
- Window attention imposes a fixed attention window W: "window attention restricts each token to attend only to previous tokens within a window size W" (Section 3.1 Experimental Settings) and Table 1 lists "Window (512)" and "Window (80)".
- Fixed/variable input resolution: Not specified (text tokens only).
- Fixed patch size: Not specified.
- Fixed number of tokens: Training uses C=2048, but evaluation varies length as "input lengths (from 2K to 8K)" are tested (Section 4.3 Results on Language Modeling).
- Fixed dimensionality (e.g., strictly 2D): Not specified.
- Padding or resizing requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length evaluated: "perform inference on samples consisting of 8192 tokens" (Section 3.3.1 Direct Extrapolation) and "input lengths (from 2K to 8K)" (Section 4.3 Results on Language Modeling).
- Context window size: "C=2048" (Section 3.1 Experimental Settings).
- Sequence length fixed or variable: Variable, as "input lengths (from 2K to 8K)" are evaluated (Section 4.3 Results on Language Modeling).
- Attention type: "Full attention means that each token can attend to all previous tokens, while window attention restricts each token to attend only to previous tokens within a window size W" (Section 3.1 Experimental Settings).
- Computational cost mechanisms: windowed attention and extensions such as "attention window extension" where "the window size also needs to be extended" and "scale the attention logits with a scaling factor  $\lambda$" (Section 4.2 Attention Window Extension).

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanisms used/compared: "variants without positional encodings (NoPE) [19] as well as variants with two different positional encodings: RoPE [7] and ALiBi [6]" (Section 3.1 Experimental Settings).
- Where it is applied: Not specified.
- Fixed vs modified: Positional encodings are compared across variants, and extension methods are applied such as "including dynamic-NTK [11] for TL-RoPE and attention scaling  $(\mathbf{q}_i \mathbf{k}_j \text{ multiplied by a scaling factor } \lambda)$  [20] for TL-NoPE" (Section 3.3.2 Context Window Extension).

## 9. Positional Encoding as a Variable

- Positional encoding is treated as a core research variable via explicit model comparisons: "we consider model variants with different positional encodings (PE) and attention mechanisms: variants without positional encodings (NoPE) [19] as well as variants with two different positional encodings: RoPE [7] and ALiBi [6]" (Section 3.1 Experimental Settings).
- Multiple positional encodings compared: Yes, as above.
- Claims that PE choice is not critical or secondary: Not claimed.

## 10. Evidence of Constraint Masking

- Model sizes: "TinyLlama-1.1B checkpoint" (Section 3.1 Experimental Settings) and "Parameters 1.3B" (Table 5 Training Details of Models).
- Dataset size: "50B tokens from RedPajama" (Section 3.1 Experimental Settings) and "Tokens 50B" (Table 5 Training Details of Models).
- Performance gains attributed to training-free architectural adjustments rather than scaling model/data: "two training-free context window extension methods" (Abstract) and "we observe that PPL decreases substantially, showing the effectiveness of our proposed methods" (Section 4.3 Results on Language Modeling).

## 11. Architectural Workarounds

- Windowed attention: "window attention restricts each token to attend only to previous tokens within a window size W" (Section 3.1 Experimental Settings) to limit attention scope.
- Positional vector replacement: "replace all the implicitly learned positional vectors with the interpolated ones, called positional vector replacement, to avoid the OOD issue" (Section 4.1 Positional Vector Replacement).
- Attention window extension: "extend the attention window size" and "scale the attention logits with a scaling factor  $\lambda$" (Section 4.2 Attention Window Extension) to interpolate positional vectors for longer contexts.
- Attention scaling baseline: "attention scaling  $(\mathbf{q}_i \mathbf{k}_j \text{ multiplied by a scaling factor } \lambda)$" (Section 3.3.2 Context Window Extension).
- Dynamic-NTK baseline for RoPE: "dynamic-NTK [11] for TL-RoPE" (Section 3.3.2 Context Window Extension).

## 12. Explicit Limitations and Non-Claims

- Limitation: "our study is mainly constrained by the use of small-scale LLMs that we trained ourselves, due to the unavailability of existing LLMs with the specific positional encodings and attention patterns required for our experiments" and "we have demonstrated the effectiveness of our proposed methods solely on our own models" (Section 7 Limitation).
- Explicit non-claims about open-world, unrestrained multi-task learning, or cross-domain transfer: Not specified.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Text-only LM datasets ("language modeling performance on the test set of PG-19" and "We subsample 32K samples with the same number of tokens from RedPajama") (Section 4.3; Section 3.1).
> - Task structure: Single primary task of language modeling/perplexity across lengths ("measure PPL across various input lengths (from 2K to 8K)") (Section 4.3).
> - Representation rigidity: Fixed training context window "C=2048" and window attention sizes "Window (512)" / "Window (80)" (Section 3.1; Table 1).
> - Model sharing vs specialization: Training-free evaluation "without fine-tuning" on pretrained variants (Introduction; Abstract).
> - Role of positional encoding: Explicitly varied across NoPE/RoPE/ALiBi ("we consider model variants with different positional encodings (PE) and attention mechanisms: variants without positional encodings (NoPE) [19] as well as variants with two different positional encodings: RoPE [7] and ALiBi [6]") (Section 3.1).

### 14. Final Classification

Classification: **Single-task, single-domain.** The experiments are language modeling evaluations on text datasets, e.g., "language modeling performance on the test set of PG-19" and inference on "We subsample 32K samples with the same number of tokens from RedPajama" with PPL analysis (Section 4.3; Section 3.1). The paper varies positional encodings and attention mechanisms but keeps the task as language modeling rather than multiple distinct tasks or modalities ("we consider model variants with different positional encodings (PE) and attention mechanisms: variants without positional encodings (NoPE) [19] as well as variants with two different positional encodings: RoPE [7] and ALiBi [6]") (Section 3.1).
