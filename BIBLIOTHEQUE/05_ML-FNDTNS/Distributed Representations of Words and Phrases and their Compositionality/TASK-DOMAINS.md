# Distributed Representations of Words and Phrases and their Compositionality (Not specified in the paper)
Source: Distributed Representations of Words and Phrases and their Compositionality.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Context word prediction (Skip-gram) | token (center word/phrase) | 0D (inferred) | Fixed (inferred) | Static (inferred) | Not specified in the paper. | token (surrounding word) | 0D (inferred) | Fixed (inferred) |
| Phrase identification (multiword expression detection) | tokens (word sequence) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Not specified in the paper. | tokens (phrases as single tokens) | 1D (t) (inferred) | Open (inferred) |
| Analogy completion (word/phrase analogies) | tokens (three words/phrases in analogy) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Not specified in the paper. | token (analogy answer word/phrase) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper centers on text-token tasks: Skip-gram context word prediction, data-driven phrase identification, and analogy completion for words and phrases. Inputs and outputs are token-level, spanning 0D single-token mappings and 1D (t) sequences (inferred), with fixed-size mappings for prediction/analogy and open-length sequences for phrase discovery (inferred). Attention is static where specified by fixed windows or rules (inferred), while state dynamics are not explicitly described.

## Evidence
### Task: Context word prediction (Skip-gram)
- "predicting the surrounding words in a sentence or a document." (Section 2 The Skip-gram Model)
- "The training objective is to learn word vector representations that are good at predicting the nearby words." (Figure 1)
- Inference: Inferred 0D fixed input/output and static attention because the objective is defined per center word with a fixed context size c ("given a sequence of training words"; "where c is the size of the training context"). (Section 2 The Skip-gram Model)

### Task: Phrase identification (multiword expression detection)
- "we first find words that appear frequently together, and infrequently in other contexts." (Section 4 Learning Phrases)
- "The bigrams with score above the chosen threshold are then used as phrases." (Section 4 Learning Phrases)
- Inference: Inferred 1D (t) sequences, open dynamics, and static attention because phrases are formed from word/bigram statistics and can grow to longer sequences ("phrases are formed based on the unigram and bigram counts"; "allowing longer phrases that consists of several words to be formed"). (Section 4 Learning Phrases)

### Task: Analogy completion (word/phrase analogies)
- "The task consists of analogies such as "Germany": "Berlin":: "France": ?" (Section 3 Empirical Results)
- "We evaluate the quality of the phrase representations using a new analogical reasoning task that involves phrases." (Section 4 Learning Phrases)
- "The goal is to compute the fourth phrase using the first three." (Table 2)
- Inference: Inferred fixed 3-to-1 token mapping and static attention from the defined analogy format ("The task consists of analogies such as "Germany": "Berlin":: "France": ?"; "The goal is to compute the fourth phrase using the first three"). (Section 3 Empirical Results; Table 2)
