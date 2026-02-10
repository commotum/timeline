# word2vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method (2014)
Source: word2vec Explained- deriving Mikolov et al.'s negative-sampling word-embedding method.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Context prediction (skip-gram) | Corpus words and context words (w, c) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Conditional probability p(c\|w) over context words | 1D (t) (inferred) | Capped (inferred) |
| Binary classification of word-context pairs (negative sampling) | Word-context pairs (w, c) from D and sampled negative pairs from D' | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Probability/label that pair came from corpus p(D=1\|w,c) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers two closely related NLP task intents: skip-gram context prediction and negative-sampling binary discrimination of word-context pairs. Both tasks operate on words and contexts extracted from linear text, so the task domain is 1D (t) (inferred). The paper’s dynamic window and fixed negative-sample count indicate capped input dynamics, while the pairwise classifier output in negative sampling is a fixed-size 0D decision (inferred). Attention is static and state is constructed (both inferred) because context selection is predefined by window/sampling rules, while learned word/context vectors are the model’s internal abstractions.

## Evidence
### Task: Context prediction (skip-gram)
- "In this model we are given a corpus of words w and their contexts c. We consider the conditional probabilities p(c|w), and given a corpus Text, the goal is to set the parameters  $\theta$  of  $p(c|w;\theta)$  so as to maximize the corpus probability:" (Section 1 The skip-gram model)
- "This section lists some peculiarities of the contexts used in the word2vec software, as reflected in the code. Generally speaking, for a sentence of n words  $w_1, \ldots, w_n$ , contexts of a word  $w_i$  comes from a window of size k around the word:  $C(w) = w_{i-k}, \ldots, w_{i-1}, w_{i+1}, \ldots, w_{i+k}$ , where k is a parameter." (Section 3 Context definitions)
- Inference: In Dimension is marked as 1D (t) because inputs are words in sentence order (Section 3). In Dynamics and Out Dynamics are marked Capped because "the parameter k denotes the *maximal* window size" and sampled window size is bounded by k (Section 3). Attention Dynamic is marked Static because context consideration is predefined by the window rule rather than runtime selection (Section 3). State Dynamic is marked Constructed because the model learns vector representations as parameters (Section 1.1: "where  $v_c$  and  $v_w \in R^d$  are vector representations for c and w respectively").

### Task: Binary classification of word-context pairs (negative sampling)
- "Consider a pair (w,c) of word and context. Did this pair come from the training data? Let's denote by p(D=1|w,c) the probability that (w,c) came from the corpus data." (Section 2 Negative Sampling)
- "This is achieved by generating the set D' of random (w,c) pairs, assuming they are all incorrect (the name \"negative-sampling\" stems from the set D' of randomly sampled negative examples)." (Section 2 Negative Sampling)
- Inference: In Dimension is marked as 1D (t) from the same word/context-in-text setup described in Sections 1 and 3. In Dynamics is marked Capped because the method uses a bounded context window and "with negative sampling of k ... for each  $(w,c) \in D$  we construct k samples" (Section 2 and Section 3). Attention Dynamic is marked Static because candidate contexts are provided by fixed window and sampling procedures, not adaptive runtime retrieval (Sections 2-3). State Dynamic is marked Constructed because "the words and contexts representations are learned jointly" (Section 2.1 Remarks). Out Dimension is 0D (inferred) and Out Dynamics is Fixed (inferred) because output is a single probability/class decision per pair: p(D=1|w,c) vs p(D=0|w,c) (Section 2).
