# Efficient Estimation of Word Representations in Vector Space (Not specified in the paper)
Source: Efficient Estimation of Word Representations in Vector Space.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Word prediction (current word from context, CBOW) | context word tokens (history and future words) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | current word token | 0D (inferred) | Fixed (inferred) |
| Context word prediction (surrounding words from current word, Skip-gram) | current word token | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | surrounding word tokens (within a range) | 1D (t) (inferred) | Capped (inferred) |
| Word relationship/analogy question answering (Semantic-Syntactic test set) | word tokens forming semantic/syntactic relationship questions | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | predicted word token (answer) | 0D (inferred) | Fixed (inferred) |
| Word similarity/relatedness evaluation (MSR Word Relatedness test set) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Sentence completion (Microsoft Sentence Completion Challenge) | sentence with one missing word and five candidate choices | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | selected word choice | 0D (inferred) | Fixed (inferred) |
| Odd-one-out word selection (out-of-the-list words) | list of word tokens | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | most distant word token | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers word-level prediction objectives (CBOW and Skip-gram) and evaluates word vectors on analogy-style semantic/syntactic questions, word relatedness, sentence completion, and odd-one-out selection. Inputs and outputs are text tokens or sentences, with inferred 1D token dimensions for sequences/lists and 0D outputs for single-word selections, while training objectives use fixed or capped context windows. Attention and state dynamics are only inferred as static/direct for the word prediction objectives and are otherwise not specified in the paper.

## Evidence
### Task: Word prediction (current word from context, CBOW)
- "log-linear classifier with four future and four history words at the input" (Section 3.1 Continuous Bag-of-Words Model)
- "training criterion is to correctly classify the current (middle) word." (Section 3.1 Continuous Bag-of-Words Model)
- Inference: Treated the context window as a fixed 1D token input with static attention and a single-word (0D) output based on the fixed window and classification description. (Section 3.1 Continuous Bag-of-Words Model)

### Task: Context word prediction (surrounding words from current word, Skip-gram)
- "use each current word as an input to a log-linear classifier with continuous projection layer" (Section 3.2 Continuous Skip-gram Model)
- "predict words within a certain range before and after the current word." (Section 3.2 Continuous Skip-gram Model)
- "use R words from history and R words from the future of the current word as correct labels." (Section 3.2 Continuous Skip-gram Model)
- Inference: Interpreted the single current word as 0D input and the surrounding-word outputs as a capped 1D token list with static attention and direct state. (Section 3.2 Continuous Skip-gram Model)

### Task: Word relationship/analogy question answering (Semantic-Syntactic test set)
- "we define a comprehensive test set that contains five types of semantic questions, and nine types of syntactic questions." (Section 4.1 Task Description)
- "compute vector X = vector(\"biggest\") - vector(\"big\") + vector(\"small\")." (Section 4 Results)
- "search in the vector space for the word closest to X measured by cosine distance, and use it as the answer to the question" (Section 4 Results)
- Inference: Treated each question as a fixed-size 1D token input and the answer as a single-word (0D) output with static attention/direct state. (Section 4 Results; Section 4.1 Task Description)

### Task: Word similarity/relatedness evaluation (MSR Word Relatedness test set)
- "We also include results on a test set introduced in [20] that focuses on syntactic similarity between words." (Section 4.3 Comparison of Model Architectures)
- "MSR Word Relatedness" (Table 3, Section 4.3 Comparison of Model Architectures)

### Task: Sentence completion (Microsoft Sentence Completion Challenge)
- "This task consists of 1040 sentences, where one word is missing in each sentence" (Section 4.5 Microsoft Research Sentence Completion Challenge)
- "the goal is to select word that is the most coherent with the rest of the sentence, given a list of five reasonable choices." (Section 4.5 Microsoft Research Sentence Completion Challenge)
- Inference: Interpreted the input as a 1D token sequence plus fixed choice list and the output as a single-word (0D) selection. (Section 4.5 Microsoft Research Sentence Completion Challenge)

### Task: Odd-one-out word selection (out-of-the-list words)
- "selecting out-of-the-list words, by computing average vector for a list of words, and finding the most distant word vector." (Section 5 Examples of the Learned Relationships)
- Inference: Treated the word list as a 1D token input and the selected word as a single-word (0D) output. (Section 5 Examples of the Learned Relationships)
