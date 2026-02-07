# On neural scaling and the quanta hypothesis (2026)
Source: On neural scaling and the quanta hypothesis.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Next-token prediction (language modeling) | tokens | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | next token | 0D (inferred) | Not specified in the paper. |
| Palindrome classification | sequence (string) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | palindrome label | 0D (inferred) | Not specified in the paper. |
| Sparse parity classification | binary strings | 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | parity label (sum modulo 2) | 0D (inferred) | Fixed (inferred) |
| Multitask sparse parity classification | binary strings with control bits + task bits | 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | parity label for selected subtask | 0D (inferred) | Fixed (inferred) |
| Mod. arithmetic | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| IPA transliterate | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Word unscramble | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Persian QA | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| TruthfulQA | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Grounded mappings | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Multi-task NLU | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Word in context | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| 3-Digit addition | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Language understanding | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Program synthesis | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Massive Multitask Language Understanding | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Logical arguments | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Sports understanding | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper focuses on language modeling as next-token prediction and includes synthetic binary-string classification tasks (sparse parity and multitask sparse parity), plus a palindrome classification example. It also lists multiple emergent-ability benchmarks (e.g., modular arithmetic, transliteration, QA, NLU, program synthesis) without specifying their input/output details. Where explicit task descriptions are given, inputs are 1D sequences and outputs are single labels or single-token predictions, with fixed-length inputs inferred for the sparse parity variants; attention and state dynamics are not specified.

## Evidence
### Task: Next-token prediction (language modeling)
- "What must be learned, to minimize loss in predicting the next token, across all the tokens in all the documents on the internet?" (Section The Quanta Hypothesis)
- "outputs a uniform distribution over the tokens" (Section The Quanta Hypothesis)
- Inference: Treated token sequences as 1D (t) inputs and the next-token output as 0D based on the wording above. (Section The Quanta Hypothesis)

### Task: Palindrome classification
- "trained to output whether the sequence seen so far is a palindrome." (Figure: palindrome classification learning curve)
- Inference: Interpreted "sequence" as a 1D (t) input and "whether" as a single-label (0D) output. (Figure: palindrome classification learning curve)

### Task: Sparse parity classification
- "Again, this is a binary classification problem on binary strings." (Section Multitask sparse parity)
- "for 100-bit strings, indices {17, 53, 89}" (Section Multitask sparse parity)
- "the label of that string is the parity (sum modulo 2) of that subset of bits." (Section Multitask sparse parity)
- Inference: Interpreted "binary strings" (including the 100-bit example) as fixed-length 1D inputs and the label as a single 0D output. (Section Multitask sparse parity)

### Task: Multitask sparse parity classification
- "We will construct a multitask version of sparse parity." (Section Multitask sparse parity)
- "we introduce extra bits to the input, which we call the \"control bits\"." (Section Multitask sparse parity)
- "for a total of  $n_{\mathrm{tasks}} + n$  bits." (Section Multitask sparse parity)
- "the label for that string is the parity of the task bits  $S_i$" (Section Multitask sparse parity)
- Inference: Treated the fixed total number of bits as fixed-length 1D inputs and the parity label as a single 0D output. (Section Multitask sparse parity)

### Task: Mod. arithmetic
- "(A) Mod. arithmetic" (Section on emergent abilities examples)

### Task: IPA transliterate
- "(B) IPA transliterate" (Section on emergent abilities examples)

### Task: Word unscramble
- "(C) Word unscramble" (Section on emergent abilities examples)

### Task: Persian QA
- "(D) Persian QA" (Section on emergent abilities examples)

### Task: TruthfulQA
- "(E) TruthfulQA" (Section on emergent abilities examples)

### Task: Grounded mappings
- "(F) Grounded mappings" (Section on emergent abilities examples)

### Task: Multi-task NLU
- "(G) Multi-task NLU" (Section on emergent abilities examples)

### Task: Word in context
- "(H) Word in context" (Section on emergent abilities examples)

### Task: 3-Digit addition
- "(Left) 3-Digit addition with GPT-3 [11]." (Fig. 2, Section 2.2)

### Task: Language understanding
- "(Middle) Language understanding with GPT-3 and Gopher [62]." (Fig. 2, Section 2.2)

### Task: Program synthesis
- "(Right) Program synthesis with Google language models [4]." (Fig. 2, Section 2.2)

### Task: Massive Multitask Language Understanding
- "Massive Multitask Language Understanding" (Figure 2 of Wei et al. (2022))

### Task: Logical arguments
- "For \"Logical arguments\" and \"Sports understanding\", the cross-entropy drops below 0.69 nats well before the error decreases." (Footnote 6)

### Task: Sports understanding
- "For \"Logical arguments\" and \"Sports understanding\", the cross-entropy drops below 0.69 nats well before the error decreases." (Footnote 6)
