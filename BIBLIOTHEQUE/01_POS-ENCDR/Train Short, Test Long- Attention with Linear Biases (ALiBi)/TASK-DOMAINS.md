# TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION (Year not specified in the paper)
Source: Train Short, Test Long- Attention with Linear Biases (ALiBi).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Next-token prediction (language modeling) | tokens (text subsequences) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | next-token probability distributions | 1D (t) (inferred) | Open (inferred) |

## Summary
This paper evaluates one task intent: language modeling as next-token prediction on text token sequences, across WikiText-103, Toronto BooksCorpus, and CC100+RoBERTa. The input and output domains are sequential tokens, which map to 1D (t) structure for both sides. The model is described as handling an arbitrary, unfixed number of input vectors for extrapolation, supporting Open dynamics in this classification. Attention is causally constrained (Static), and the task behavior is a reactive next-token mapping (Direct state).

## Evidence
### Task: Next-token prediction (language modeling)
- "A transformer LM receives a list of tokens and outputs a probability distribution representing its prediction for the next token." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP)
- "During both training and perplexity evaluation (i.e., scoring a fixed sequence), many predictions can be calculated at once; this is done using a \"causal mask\" that ensures each position's prediction is influenced only by tokens to its left." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP)
- "**Extrapolation During Inference** Formally, the functions that define a transformer layer are agnostic to input length;<sup>3</sup> they map from some arbitrary, unfixed number of input vectors to the same number of output vectors." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP, Extrapolation During Inference)
- Inference: `In Dimension` and `Out Dimension` are `1D (t)` because the paper defines the task over ordered token lists and left-to-right token predictions. `In Dynamics` and `Out Dynamics` are `Open` because the layer interface is described as accepting an "arbitrary, unfixed number of input vectors." `Attention Dynamic` is `Static` because a causal mask predefines what each position can attend to. `State Dynamic` is `Direct` because the task is next-token prediction from the current token subsequence (reactive mapping, with no persistent constructed state described).
