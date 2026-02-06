# End-To-End Memory Networks (Not specified in the paper)
Source: End-To-End Memory Networks (MemN2N).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| question answering | statements (sentences) and question sentence | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | answer word(s) | 0D (inferred) | Fixed (inferred) |
| language modeling (next-word prediction) | previous words in a text sequence | 1D (t) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | next word | 0D (inferred) | Fixed (inferred) |

## Summary
The paper applies the model to synthetic text-based question answering and to word-level language modeling. Inputs are sequences of sentences or word histories, with outputs as single-word answers or next-word predictions, implying 1D textual inputs and 0D label outputs. Both tasks use memory-based attention over stored text and a constructed internal memory state, with bounded input sizes via dataset limits (QA) or fixed context windows (language modeling).

## Evidence
### Task: question answering
- "We perform experiments on the synthetic QA tasks defined in [22] (using version 1.1 of the dataset)." (Section 4 Synthetic Question and Answering Experiments)
- "A given QA task consists of a set of statements, followed by a question" (Section 4 Synthetic Question and Answering Experiments)
- "whose answer is typically a single word (in a few tasks, answers are a set of words)." (Section 4 Synthetic Question and Answering Experiments)
- "a set of I sentences  $\\{x_i\\}$  where  $I \\leq 320$ ; a question sentence q and answer a." (Section 4 Synthetic Question and Answering Experiments)
- "Hence, the model must deduce for itself at training and test time which sentences are relevant and which are not." (Section 4 Synthetic Question and Answering Experiments)
- "Our model takes a discrete set of inputs  $x_1, ..., x_n$  that are to be stored in the memory" (Section 2 Approach)
- Inference: In Dimension (1D (t)), In Dynamics (Capped), Attention Dynamic (Dynamic), State Dynamic (Constructed), Out Dimension (0D), and Out Dynamics (Fixed) are inferred from the sequential sentences/words, the explicit cap on I, the need to select relevant sentences, memory storage, and single-word answers.

### Task: language modeling (next-word prediction)
- "The goal in language modeling is to predict the next word in a text sequence given the previous words x." (Section 5 Language Modeling Experiments)
- "Thus the previous N words in the sequence (including the current) are embedded into memory separately." (Section 5 Language Modeling Experiments)
- "The output softmax predicts which word in the vocabulary (of size V) is next in the sequence." (Section 5 Language Modeling Experiments)
- "we compute the match between u and each memory  $m_i$  by taking the inner product followed by a softmax" (Section 2.1 Single Layer)
- Inference: In Dimension (1D (t)), In Dynamics (Fixed), Attention Dynamic (Dynamic), State Dynamic (Constructed), Out Dimension (0D), and Out Dynamics (Fixed) are inferred from the word sequence framing, fixed N-word context window, memory-based softmax over stored words, and next-word prediction.
