# Exploring Context Window of Large Language Models via Decomposed Positional Vectors (Not specified in the paper.)
Source: Exploring Context Window of LLMs.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language modeling | Tokens (text sequence) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Token prediction probabilities | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates a single text-only task—language modeling—using decoder-only Transformers and perplexity as the metric. Inputs are token sequences and outputs are token prediction probabilities, implying a 1D temporal structure (inferred). The interface is bounded by a context window and attention is predefined as full or fixed window, so dynamics are capped and attention static (inferred).

## Evidence
### Task: Language modeling
- "we evaluate language modeling performance on the test set of PG-19 [22]." (Section 4.3 Results on Language Modeling)
- "given an input sequence s of T tokens" (Section 2 Background)
- "projected into the logits, which will be used to generate the prediction probability for each token" (Section 2 Background)
- "with a context window C=2048" (Section 3.1 Experimental Settings)
- "Full attention means that each token can attend to all previous tokens" (Section 3.1 Experimental Settings)
- "window attention restricts each token to attend only to previous tokens within a window size W." (Section 3.1 Experimental Settings)
- Inference: Marked In/Out Dimension as 1D (t) and In/Out Dynamics as Capped because the task uses a token sequence and a fixed context window ("given an input sequence s of T tokens"; "with a context window C=2048"). (Section 2 Background; Section 3.1 Experimental Settings)
- Inference: Marked Attention Dynamic as Static because attention scope is predefined as full or window attention ("Full attention means that each token can attend to all previous tokens"; "window attention restricts each token to attend only to previous tokens within a window size W."). (Section 3.1 Experimental Settings)
- Inference: Marked State Dynamic as Direct because the model maps input sequences to token logits without any described constructed external state ("projected into the logits, which will be used to generate the prediction probability for each token"). (Section 2 Background)

## CSV Output (required)
CSV written to: "/home/jake/Developer/timeline/BIBLIOTHEQUE/01_POS-ENCDR/Exploring Context Window of LLMs/.TASK-DOMAINS.csv.tmp.c53843a1dd484251a1f2a71ca008d0a0"
