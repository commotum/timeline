# A Neural Probabilistic Language Model (Not specified in the paper)
Source: A Neural Probabilistic Language Model.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next-word prediction (language modeling) | word sequence context (previous words) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | next-word probabilities over vocabulary | 0D (inferred) | Fixed (inferred) |

## Summary
The paper frames statistical language modeling as learning the probability of word sequences and predicts the next word from prior context. The modeled inputs are 1D word sequences with a fixed-length context window (inferred), producing a single next-word probability distribution over a fixed vocabulary. The attention is static and the state is direct (both inferred) based on the fixed conditional formulation.

## Evidence
### Task: next-word prediction (language modeling)
- "A goal of statistical language modeling is to learn the joint probability function of sequences of words in a language." (Abstract)
- "A statistical model of language can be represented by the conditional probability of the next word given all the previous ones" (Section 1. Introduction)
- "The objective is to learn a good model f(w_t, \cdots, w_{t-n+1}) = \hat{P}(w_t|w_1^{t-1})" (Section 2. A Neural Model)
- "The training set is a sequence w_1 \cdots w_T of words w_t in V, where the vocabulary V is a large but finite set." (Section 2. A Neural Model)
- Inference: In Dimension is 1D (t), In Dynamics is Fixed, Attention is Static, State is Direct, and Out Dimension/Out Dynamics are 0D/Fixed because the model maps a fixed context window of words to a single next-word distribution over a finite vocabulary. (inferred from Section 2. A Neural Model)
